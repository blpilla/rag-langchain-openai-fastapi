import pytest
from unittest.mock import MagicMock, patch
from src.rag_engine import RAGEngine
from src.vector_db import VectorDB

@pytest.fixture
def mock_vector_db():
    mock = MagicMock(spec=VectorDB)
    mock.index = MagicMock()  # Adiciona o atributo 'index'
    mock.documents = []  # Adiciona o atributo 'documents'
    return mock

@pytest.fixture
def mock_openai():
    with patch('src.rag_engine.OpenAI') as mock:
        yield mock

@pytest.fixture
def mock_embeddings():
    with patch('src.rag_engine.OpenAIEmbeddings') as mock:
        mock.return_value.embed_query = MagicMock()  # Adiciona o método 'embed_query'
        yield mock

@pytest.fixture
def mock_faiss():
    with patch('src.rag_engine.FAISS') as mock:
        yield mock

@pytest.fixture
def mock_retrieval_qa():
    with patch('src.rag_engine.RetrievalQA') as mock:
        yield mock

def test_rag_engine_initialization(mock_vector_db, mock_openai, mock_embeddings, mock_faiss, mock_retrieval_qa):
    rag_engine = RAGEngine(mock_vector_db)
    
    assert mock_openai.called, "OpenAI should be initialized"
    assert mock_embeddings.called, "OpenAIEmbeddings should be initialized"
    assert mock_faiss.called, "FAISS should be initialized"
    assert mock_retrieval_qa.from_chain_type.called, "RetrievalQA should be initialized"

def test_rag_engine_query(mock_vector_db, mock_openai, mock_embeddings, mock_faiss, mock_retrieval_qa):
    mock_qa_chain = MagicMock()
    mock_qa_chain.run.return_value = "Resposta simulada"
    mock_retrieval_qa.from_chain_type.return_value = mock_qa_chain

    rag_engine = RAGEngine(mock_vector_db)
    result = rag_engine.query("Pergunta de teste?")

    assert result == "Resposta simulada", "A query deve retornar a resposta do QA chain"
    mock_qa_chain.run.assert_called_once_with("Pergunta de teste?")

def test_rag_engine_error_handling(mock_vector_db, mock_openai, mock_embeddings, mock_faiss, mock_retrieval_qa):
    mock_qa_chain = MagicMock()
    mock_qa_chain.run.side_effect = Exception("Erro simulado")
    mock_retrieval_qa.from_chain_type.return_value = mock_qa_chain

    rag_engine = RAGEngine(mock_vector_db)
    
    with pytest.raises(Exception, match="Erro simulado"):
        rag_engine.query("Pergunta que gera erro")

@patch.dict('os.environ', {'OPENAI_API_KEY': ''}, clear=True)
def test_rag_engine_missing_api_key(mock_vector_db):
    with pytest.raises(ValueError, match="OPENAI_API_KEY não encontrada nas variáveis de ambiente"):
        RAGEngine(mock_vector_db)