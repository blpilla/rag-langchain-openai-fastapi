import pytest
from unittest.mock import MagicMock, patch
from src.rag_engine import RAGEngine

@pytest.fixture
def mock_vector_db():
    """
    Cria um mock de um banco de dados vetorial para testes.

    Retorna:
        mock: Um objeto mock que simula o comportamento de um banco de dados vetorial.
    """
    mock = MagicMock()
    mock.get_vector_store.return_value = MagicMock()
    return mock

@pytest.fixture
def mock_openai():
    """
    Cria um mock do OpenAI para testes.
    
    Retorna:
        mock: Um objeto mock que simula o comportamento do OpenAI.
    """
    with patch('src.rag_engine.OpenAI') as mock:
        yield mock

@pytest.fixture
def mock_embeddings():
    """
    Cria um mock das embeddings do OpenAI para testes.
    
    Retorna:
        mock: Um objeto mock que simula o comportamento das embeddings do OpenAI.
    """
    with patch('src.rag_engine.OpenAIEmbeddings') as mock:
        yield mock

@pytest.fixture
def mock_retrieval_qa():
    """
    Cria um mock do RetrievalQA para testes.

    Retorna:
        mock: Um objeto mock que simula o comportamento do RetrievalQA.
    """
    with patch('src.rag_engine.RetrievalQA') as mock:
        yield mock

def test_rag_engine_initialization(mock_vector_db, mock_openai, mock_embeddings, mock_retrieval_qa):
    # Testa a inicialização correta do RAGEngine
    rag_engine = RAGEngine(mock_vector_db)
    
    assert mock_openai.called, "OpenAI deveria ser inicializado"
    assert mock_embeddings.called, "OpenAIEmbeddings deveria ser inicializado"
    assert mock_retrieval_qa.from_chain_type.called, "RetrievalQA deveria ser inicializado"
    assert mock_vector_db.get_vector_store.called, "VectorDB deveria ser consultado"

def test_rag_engine_query(mock_vector_db, mock_openai, mock_embeddings, mock_retrieval_qa):
    # Testa a funcionalidade de consulta do RAGEngine
    mock_qa_chain = MagicMock()
    mock_qa_chain.invoke.return_value = {
        "result": "Test answer",
        "source_documents": [
            MagicMock(metadata={"source": "doc1"}, page_content="content1"),
            MagicMock(metadata={"source": "doc2"}, page_content="content2")
        ]
    }
    mock_retrieval_qa.from_chain_type.return_value = mock_qa_chain

    rag_engine = RAGEngine(mock_vector_db)
    result = rag_engine.query("Test question")

    # Verifica se a resposta tem a estrutura esperada
    assert "answer" in result
    assert "sources" in result
    assert len(result["sources"]) == 2
    assert result["sources"][0] == {
        "title": "doc1",
        "content": "content1",
        "metadata": {"source": "doc1"}
    }
    assert result["sources"][1] == {
        "title": "doc2",
        "content": "content2",
        "metadata": {"source": "doc2"}
    }

def test_rag_engine_query_no_documents(mock_vector_db, mock_openai, mock_embeddings, mock_retrieval_qa):
    # Testa o comportamento quando não há documentos disponíveis
    mock_vector_db.get_vector_store.return_value = None
    rag_engine = RAGEngine(mock_vector_db)
    result = rag_engine.query("Test question")

    assert "answer" in result
    assert "Desculpe, não há documentos para responder à sua pergunta." in result["answer"]
    assert result["sources"] == []