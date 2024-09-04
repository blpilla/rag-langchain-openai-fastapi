from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List
from src.document_processor import DocumentProcessor, DocumentProcessingError
from src.vector_db import VectorDB
from src.rag_engine import RAGEngine
import logging
import os

app = FastAPI()

# Configura o logging para monitorar a execução do aplicativo
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Inicializa os componentes principais do sistema
document_processor = DocumentProcessor()
vector_db = VectorDB(persist_directory="./persistent_vector_db")
rag_engine = RAGEngine(vector_db)

class Query(BaseModel):
    question: str

@app.post("/upload_documents")
async def upload_documents(files: List[UploadFile] = File(...)):
    """
    Upload de documentos para processamento e armazenamento no banco de dados vetorial.

    Parâmetros:
        files (List[UploadFile]): Lista de arquivos a serem carregados e processados.

    Retorno:
        dict: Mensagem de sucesso com o número de documentos carregados e processados.

    Exceções:
        DocumentProcessingError: Erro ao processar os documentos.
        Exception: Erro ao fazer upload e processar os documentos.
    """
    try:
        processed_documents = []
        for file in files:
            content = await file.read()
            logger.info(f"Processando arquivo: {file.filename}")
            # Processa cada arquivo e extrai seu conteúdo
            processed_segments = document_processor.process_file(content, file.filename)
            logger.info(f"Segmentos processados: {len(processed_segments)}")
            # Adiciona os segmentos processados ao dicionário de documentos
            processed_documents.extend(processed_segments)
        
        logger.info(f"Total de segmentos processados: {len(processed_documents)}")
        # Adiciona os segmentos processados ao banco de dados vetorial
        for segment in processed_documents:
            vector_db.add([segment["content"]], [segment["metadata"]])
        
        # Reinicializa o motor RAG com o banco de dados atualizado
        global rag_engine
        rag_engine = RAGEngine(vector_db)
        logger.info("RAGEngine reinicializado com novos documentos")
        # Retorna uma mensagem de sucesso
        return {"message": f"{len(files)} documentos carregados, processados e armazenados com sucesso"}
    # Trata erros de processamento de documentos
    except DocumentProcessingError as e:
        logger.error(f"Erro ao processar documentos: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Erro ao fazer upload e processar documentos: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
async def query(query: Query):
    """
    Processa uma consulta utilizando o motor RAG.

    Parâmetros:
        query (Query): Objeto contendo a pergunta a ser processada.

    Retorna:
        dict: Um dicionário contendo a pergunta, a resposta processada e as fontes utilizadas.
            - question (str): A pergunta processada.
            - answer (str): A resposta processada.
            - sources (list): Uma lista de dicionários contendo as fontes utilizadas.

    Lança:
        HTTPException: Se ocorrer um erro durante o processamento da consulta.
    """
    try:
        logger.info(f"Recebida consulta: {query.question}")
        # Processa a consulta usando o motor RAG
        response = rag_engine.query(query.question)
        logger.info(f"Resposta gerada: {response}")
        # Retorna a resposta processada
        return {
            "question": query.question,
            "answer": response["answer"],
            "sources": response["sources"]
        }
    except Exception as e:
        logger.error(f"Erro ao processar consulta: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/vector_db_status")
async def vector_db_status():
    """
    Retorna o status atual do banco de dados vetorial.
    
    Parâmetros:
        None
    
    Retorna:
        dict: Um dicionário contendo o total de documentos no banco de dados vetorial e um booleano indicando se o banco de dados está vazio.
    """
    return {
        "total_documents": vector_db.vector_store.index.ntotal if vector_db.vector_store else 0,
        "is_empty": vector_db.vector_store is None or vector_db.vector_store.index.ntotal == 0
    }