import pytest
from src.vector_db import VectorDB
import os
import shutil

@pytest.fixture(autouse=True)
def clean_vector_db():
    """
    Fixture que limpa o diretório de persistência do banco de dados vetorial antes e após cada teste.
    Garante que o diretório esteja vazio para evitar interferência entre testes.
    """
    # Remove o diretório de persistência antes de cada teste
    if os.path.exists("./vector_db"):
        shutil.rmtree("./vector_db")
    yield
    # Limpa novamente após o teste
    if os.path.exists("./vector_db"):
        shutil.rmtree("./vector_db")

@pytest.fixture
def vector_db():
    """
    Fixture que cria uma instância do banco de dados vetorial VectorDB.
    
    Parâmetros:
    Nenhum
    
    Retorno:
    Uma instância do banco de dados vetorial VectorDB com o diretório de persistência definido como "./vector_db".
    """
    return VectorDB(persist_directory="./vector_db")

def test_add_and_search(vector_db):
    # Testa a adição de documentos e a busca
    documents = ["This is a test document", "Another test document"]
    metadatas = [{"source": "test1"}, {"source": "test2"}]
    vector_db.add(documents, metadatas)
    
    results = vector_db.search("test document")
    # Verifica se a busca retorna resultados
    assert len(results) > 0
    # Verifica se o resultado contém o texto esperado
    assert "test document" in results[0][0]

def test_empty_search(vector_db):
    # Testa a busca em um banco de dados vazio
    results = vector_db.search("nonexistent document")
    assert len(results) == 0, "Busca em DB vazio deve retornar lista vazia"

def test_error_handling(vector_db):
    # Testa o tratamento de erros para entradas inválidas
    with pytest.raises(ValueError):
        vector_db.add(None, None)

    with pytest.raises(ValueError):
        vector_db.add([], None)

    with pytest.raises(ValueError):
        vector_db.add(["test"], [])

    with pytest.raises(ValueError):
        vector_db.add("not a list", "not a list")