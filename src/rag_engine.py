from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain.chains import RetrievalQA
import os
from dotenv import load_dotenv
import logging
import traceback

# Configuração do logging para monitoramento e debugging
logger = logging.getLogger(__name__)

# Carrega variáveis de ambiente do arquivo .env
load_dotenv()

class RAGEngine:
    def __init__(self, vector_db):
        """
        Inicializa o RAGEngine com um banco de dados vetorial.

        Este método configura os componentes necessários para o sistema RAG,
        incluindo o modelo de linguagem, embeddings e o chain de pergunta e resposta.

        Parâmetros:
            vector_db: Um objeto que representa o banco de dados vetorial.

        Lança:
            ValueError: Se a chave da API do OpenAI não for encontrada nas variáveis de ambiente.

        Retorna:
            None
        """
        # Obtém a chave da API do OpenAI das variáveis de ambiente
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY não encontrada nas variáveis de ambiente")
        
        # Inicializa o modelo de linguagem OpenAI
        self.llm = OpenAI(api_key=api_key)
        
        # Inicializa o modelo de embeddings OpenAI
        self.embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        
        # Obtém o armazenamento de vetores do banco de dados vetorial
        self.vector_store = vector_db.get_vector_store()
        
        # Verifica se o banco de dados vetorial está vazio
        if self.vector_store is None:
            logger.warning("VectorDB está vazio. O RAGEngine não pode ser inicializado completamente.")
            self.qa_chain = None
        else:
            # Cria a cadeia de pergunta e resposta (QA Chain)
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",  # Usa o método "stuff" para combinar documentos
                retriever=self.vector_store.as_retriever(),
                return_source_documents=True,  # Retorna os documentos fonte usados
                verbose=True  # Ativa logs detalhados para debugging
            )

    def query(self, question):
        """
        Processa uma consulta utilizando o QA Chain.

        Este método toma uma pergunta como entrada, usa o QA Chain para processá-la,
        e retorna uma resposta junto com as fontes relevantes utilizadas.

        Parâmetros:
            question (str): A pergunta a ser processada.

        Retorna:
            dict: Um dicionário contendo a resposta processada e as fontes utilizadas.
                - answer (str): A resposta processada.
                - sources (list): Uma lista de dicionários contendo as fontes utilizadas.
                    - title (str): O título da fonte.
                    - content (str): O conteúdo da fonte.
                    - metadata (dict): Os metadados da fonte.

        Lança:
            Exception: Se ocorrer um erro durante o processamento da consulta.
        """
        try:
            # Verifica se o QA Chain foi inicializado corretamente
            if self.qa_chain is None:
                logger.warning("QA Chain não inicializada")
                return {
                    "answer": "Desculpe, não há documentos para responder à sua pergunta.",
                    "sources": []
                }
            
            # Loga a consulta para fins de debugging
            logger.info(f"Processando consulta: {question}")
            
            # Invoca o QA Chain para processar a consulta
            # Usa 'invoke' em vez de chamar diretamente para compatibilidade com versões mais recentes do LangChain
            result = self.qa_chain.invoke({"query": question})
            logger.info(f"Resposta bruta do QA Chain: {result}")
            
            # Extrai a resposta gerada
            answer = result['result']
            
            # Obtém os documentos fonte usados para gerar a resposta
            # Se não houver documentos fonte, usa uma lista vazia
            source_documents = result.get('source_documents', [])
            
            logger.info(f"Número de documentos fonte: {len(source_documents)}")
            
            # Processa os documentos fonte para extrair informações relevantes
            sources = []
            for doc in source_documents:
                # Extrai metadados do documento, se disponíveis
                metadata = doc.metadata if hasattr(doc, 'metadata') else {}
                logger.info(f"Metadados do documento: {metadata}")
                
                # Cria um dicionário com informações da fonte
                sources.append({
                    "title": metadata.get("source", "Título não disponível"),
                    "content": doc.page_content if hasattr(doc, 'page_content') else "Conteúdo não disponível",
                    "metadata": metadata
                })
            
            # Loga as fontes processadas para debugging
            logger.info(f"Fontes processadas: {sources}")
            
            # Retorna um dicionário com a resposta e as fontes
            return {
                "answer": answer,
                "sources": sources
            }
        except Exception as e:
            # Loga qualquer erro que ocorra durante o processamento
            logger.error(f"Erro ao processar consulta: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            # Re-lança a exceção para ser tratada em um nível superior
            raise