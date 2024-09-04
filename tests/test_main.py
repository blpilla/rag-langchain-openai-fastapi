from fastapi.testclient import TestClient
from main import app
import os
import shutil

client = TestClient(app)

def setup_module(module):
    """
    Configura o módulo antes da execução dos testes, limpando o diretório de persistência se existir.
    
    Parâmetros:
        module (objeto): O módulo de teste atual.
    
    Retorno:
        None
    """
    if os.path.exists("./persistent_vector_db"):
        shutil.rmtree("./persistent_vector_db")

# Testa o upload de arquivos
def test_upload_documents():
    with open("test_document.txt", "w") as f:
        f.write("This is a test document for upload.")
    
    with open("test_document.txt", "rb") as f:
        response = client.post("/upload_documents", files={"files": ("test_document.txt", f)})
    
    assert response.status_code == 200
    assert "documentos carregados" in response.json()["message"]

    # Limpa o arquivo de teste após o uso
    os.remove("test_document.txt")

# Testa a consulta da API
def test_query():
    response = client.post("/query", json={"question": "What is this document about?"})
    assert response.status_code == 200
    assert "question" in response.json()
    assert "answer" in response.json()
    assert "sources" in response.json()

# Testa o tratamento de erros
def test_error_handling():
    response = client.post("/query", json={"invalid": "data"})
    assert response.status_code == 422  # Unprocessable Entity

# Limpa o diretório de persistência após os testes
def teardown_module(module):
    if os.path.exists("./persistent_vector_db"):
        shutil.rmtree("./persistent_vector_db")