from langchain_openai import OpenAI
from langchain.chains import RetrievalQA
import os
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)

load_dotenv()

class RAGEngine:
    def __init__(self, vector_db):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY não encontrada nas variáveis de ambiente")
        
        self.llm = OpenAI(api_key=api_key)
        self.vector_store = vector_db.get_vector_store()
        
        if self.vector_store is None:
            logger.warning("VectorDB está vazio. O RAGEngine não pode ser inicializado completamente.")
            self.qa_chain = None
        else:
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.vector_store.as_retriever()
            )

    def query(self, question):
        try:
            if self.qa_chain is None:
                return "Desculpe, não há documentos para responder à sua pergunta."
            
            logger.info(f"Processando consulta: {question}")
            result = self.qa_chain.invoke(question)
            logger.info(f"Resposta gerada: {result}")
            return result['result'] if isinstance(result, dict) and 'result' in result else result
        except Exception as e:
            logger.error(f"Erro ao processar consulta: {str(e)}")
            raise