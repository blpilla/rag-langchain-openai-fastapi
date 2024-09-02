import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from main import app, document_processor

client = TestClient(app)

@pytest.fixture
def mock_document_processor():
    with patch('main.document_processor') as mock:
        yield mock

@pytest.fixture
def mock_text_to_vector():
    with patch('main.text_to_vector') as mock:
        yield mock

@pytest.fixture
def mock_vector_db():
    with patch('main.vector_db') as mock:
        yield mock

@pytest.fixture
def mock_rag_engine():
    with patch('main.rag_engine') as mock:
        yield mock

def test_process_document(mock_document_processor, mock_text_to_vector, mock_vector_db):
    mock_document_processor.process.return_value = ["Segmento 1", "Segmento 2"]
    mock_text_to_vector.convert.return_value = [0.1, 0.2, 0.3]
    response = client.post("/process_document", json={"content": "Documento de teste"})
    assert response.status_code == 200
    assert response.json() == {"message": "Documento processado e armazenado com sucesso"}

def test_process_document_error(mock_document_processor):
    mock_document_processor.process.side_effect = Exception("Erro simulado")
    response = client.post("/process_document", json={"content": "Documento de teste"})
    assert response.status_code == 500
    assert "Erro simulado" in response.json()["detail"]

def test_query(mock_rag_engine):
    mock_rag_engine.query.return_value = "Resposta simulada"
    response = client.post("/query", json={"question": "Pergunta de teste?"})
    assert response.status_code == 200
    assert response.json() == {
        "question": "Pergunta de teste?",
        "answer": "Resposta simulada"
    }

def test_query_error(mock_rag_engine):
    mock_rag_engine.query.side_effect = Exception("Erro simulado")
    response = client.post("/query", json={"question": "Pergunta de teste?"})
    assert response.status_code == 500
    assert "Erro simulado" in response.json()["detail"]

def test_invalid_json():
    response = client.post("/process_document", json={"invalid": "data"})
    assert response.status_code == 422  # Unprocessable Entity

    response = client.post("/query", json={"invalid": "data"})
    assert response.status_code == 422  # Unprocessable Entity