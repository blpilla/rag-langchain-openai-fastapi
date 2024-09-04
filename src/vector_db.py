from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv
import logging
import pickle
from src.text_preprocessor import TextPreprocessor

# Configuração do logging para monitoramento e debugging
logger = logging.getLogger(__name__)

# Carrega variáveis de ambiente do arquivo .env
load_dotenv()

class VectorDB:
    def __init__(self, persist_directory="./vector_db"):
        """
        Inicializa um objeto VectorDB.

        Este método configura o banco de dados vetorial, incluindo o modelo de embeddings,
        o armazenamento de vetores e o preprocessador de texto.

        Parâmetros:
            persist_directory (str): O diretório onde o banco de dados vetorial será armazenado.
                                     O padrão é "./vector_db".

        Lança:
            ValueError: Se a chave da API do OpenAI não for encontrada nas variáveis de ambiente.

        Retorna:
            None
        """
        # Obtém a chave da API do OpenAI das variáveis de ambiente
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY não encontrada nas variáveis de ambiente")
        
        # Inicializa o modelo de embeddings da OpenAI
        self.embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        self.vector_store = None
        self.persist_directory = persist_directory

        # Inicializa o pre-processador com o idioma em português
        self.preprocessor = TextPreprocessor(language='portuguese')
        
        # Tenta carregar um banco de dados existente
        self.load()

    def add(self, texts, metadatas):
        """
        Adiciona uma lista de textos e metadados ao banco de dados vetorial.

        Este método pré-processa os textos, cria ou atualiza o armazenamento de vetores,
        e persiste os dados.

        Parâmetros:
            texts (list): Lista de textos a serem adicionados.
            metadados (list): Lista de metadados correspondentes aos textos.

        Retorno:
            None

        Exceções:
            ValueError: Se texts ou metadados forem None, ou se não forem listas, ou se tiverem tamanhos diferentes.
            Exception: Se ocorrer um erro durante a adição ao banco de dados.
        """
        # Valida as entradas antes de processar
        if texts is None or metadatas is None:
            raise ValueError("Textos e metadados não podem ser None")
        
        if not isinstance(texts, list) or not isinstance(metadatas, list):
            raise ValueError("Textos e metadados devem ser listas")
        
        if len(texts) != len(metadatas):
            raise ValueError("O número de textos deve ser igual ao número de metadados")

        try:
            logger.info(f"Adicionando {len(texts)} textos ao VectorDB")
            logger.info(f"Metadados: {metadatas}")
            
            # Pré-processa os textos
            preprocessed_texts = [self.preprocessor.preprocess(text) for text in texts]
            
            if self.vector_store is None:
                # Cria um novo FAISS VectorStore se ainda não existir
                logger.info("Inicializando novo FAISS VectorStore")
                self.vector_store = FAISS.from_texts(preprocessed_texts, self.embeddings, metadatas=metadatas)
            else:
                # Adiciona ao FAISS VectorStore existente
                logger.info("Adicionando a FAISS VectorStore existente")
                self.vector_store.add_texts(preprocessed_texts, metadatas=metadatas)
            
            logger.info(f"Total de documentos após adição: {self.vector_store.index.ntotal}")
            
            # Persiste o banco de dados após a adição
            self.save()
        except Exception as e:
            logger.error(f"Erro ao adicionar ao VectorDB: {str(e)}")
            raise

    def search(self, query, k=5):
        """
        Realiza uma busca por similaridade no banco de dados vetorial.

        Este método pré-processa a query e utiliza o FAISS para encontrar documentos similares.

        Parâmetros:
            query (str): A query a ser pesquisada.
            k (int, opcional): O número máximo de resultados a serem retornados. O padrão é 5.

        Retorna:
            list: Uma lista de tuplas contendo o conteúdo da página, os metadados e a pontuação do documento.
                Cada tupla tem o seguinte formato:
                    - O conteúdo da página (str)
                    - Os metadados do documento (dict)
                    - A pontuação do documento (float)

        Exceções:
            None
        """
        # Verifica se o FAISS VectorStore foi inicializado corretamente
        if self.vector_store is None or self.vector_store.index.ntotal == 0:
            return []
        
        # Pré-processa a query
        preprocessed_query = self.preprocessor.preprocess(query)
        
        # Realiza a busca por similaridade no FAISS com a pergunta pre-processada
        results = self.vector_store.similarity_search_with_score(preprocessed_query, k=k)
        return [(doc.page_content, doc.metadata, score) for doc, score in results]

    def get_vector_store(self):
        """
        Retorna o armazenamento de vetores atual.

        Este método é utilizado para acessar o FAISS VectorStore diretamente.

        Parâmetros:
            None

        Retorna:
            FAISS: O armazenamento de vetores atual.
        """
        return self.vector_store

    def save(self):
        """
        Persiste o banco de dados vetorial em disco.

        Este método salva o FAISS VectorStore e informações adicionais necessárias
        para reconstruir o banco de dados em futuras sessões.

        Parâmetros:
            None

        Retorna:
            None
        """
        # Salva o FAISS VectorStore em disco
        if self.vector_store:
            logger.info(f"Salvando VectorDB em {self.persist_directory}")
            self.vector_store.save_local(self.persist_directory)
            
            # Salva o índice para docstore_id separadamente
            index_to_docstore_id = self.vector_store.index_to_docstore_id
            with open(os.path.join(self.persist_directory, "index_to_docstore_id.pkl"), "wb") as f:
                pickle.dump(index_to_docstore_id, f)
            
            logger.info("VectorDB salvo com sucesso")

    def load(self):
        """
        Carrega o banco de dados vetorial do disco, se existir.

        Este método tenta carregar um FAISS VectorStore previamente salvo.
        Se o carregamento falhar, inicializa um novo VectorStore vazio.

        Parâmetros:
            None

        Retorno:
            None
        """
        if os.path.exists(self.persist_directory):
            logger.info(f"Carregando VectorDB de {self.persist_directory}")
            try:
                self.vector_store = FAISS.load_local(
                    self.persist_directory, 
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                
                # Carrega o índice para docstore_id
                index_to_docstore_id_path = os.path.join(self.persist_directory, "index_to_docstore_id.pkl")
                if os.path.exists(index_to_docstore_id_path):
                    with open(index_to_docstore_id_path, "rb") as f:
                        self.vector_store.index_to_docstore_id = pickle.load(f)
                
                logger.info(f"VectorDB carregado com {self.vector_store.index.ntotal} documentos")
            except Exception as e:
                logger.error(f"Erro ao carregar VectorDB: {str(e)}")
                logger.info("Inicializando um novo VectorDB")
                self.vector_store = None
        else:
            logger.info("Nenhum VectorDB existente encontrado")