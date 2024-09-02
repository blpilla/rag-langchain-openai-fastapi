from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.document_processor import DocumentProcessor
from src.vector_db import VectorDB
from src.rag_engine import RAGEngine
import logging

app = FastAPI()

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Inicialização dos componentes
document_processor = DocumentProcessor()
vector_db = VectorDB()
rag_engine = RAGEngine(vector_db)

class Document(BaseModel):
    content: str

class Query(BaseModel):
    question: str

@app.post("/process_document")
async def process_document(document: Document):
    try:
        segments = document_processor.process(document.content)
        vector_db.add(segments)
        global rag_engine
        rag_engine = RAGEngine(vector_db)  # Reinicialize o RAGEngine com o VectorDB atualizado
        return {"message": "Documento processado e armazenado com sucesso"}
    except Exception as e:
        logger.error(f"Erro ao processar documento: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
async def query(query: Query):
    try:
        logger.info(f"Recebida consulta: {query.question}")
        response = rag_engine.query(query.question)
        logger.info(f"Resposta gerada: {response}")
        return {
            "question": query.question,
            "answer": response
        }
    except Exception as e:
        logger.error(f"Erro ao processar consulta: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))