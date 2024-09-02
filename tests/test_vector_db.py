import pytest
import numpy as np
from src.vector_db import VectorDB

@pytest.fixture
def vector_db():
    return VectorDB(dimension=3)  # Usando dimensão 3 para simplificar os testes

def test_add_and_search(vector_db):
    vector1 = [1.0, 0.0, 0.0]
    vector2 = [0.0, 1.0, 0.0]
    vector3 = [0.0, 0.0, 1.0]
    
    vector_db.add(vector1, "documento1")
    vector_db.add(vector2, "documento2")
    vector_db.add(vector3, "documento3")
    
    results = vector_db.search([1.0, 0.1, 0.1], k=2)
    assert results[0] == "documento1", "O primeiro resultado deve ser o documento mais próximo"
    assert len(results) == 2, "Deve retornar 2 resultados quando k=2"

def test_empty_db_search(vector_db):
    results = vector_db.search([1.0, 0.0, 0.0], k=1)
    assert len(results) == 0, "Busca em DB vazio deve retornar lista vazia"

def test_search_with_k_greater_than_db_size(vector_db):
    vector_db.add([1.0, 0.0, 0.0], "documento1")
    results = vector_db.search([1.0, 0.0, 0.0], k=5)
    assert len(results) == 1, "Deve retornar todos os documentos disponíveis, mesmo que k seja maior"

def test_add_invalid_vector(vector_db):
    with pytest.raises(ValueError, match="Vector dimension mismatch"):
        vector_db.add([1.0, 0.0], "documento_invalido")  # Vetor com dimensão errada

def test_search_invalid_vector(vector_db):
    with pytest.raises(ValueError, match="Query vector dimension mismatch"):
        vector_db.search([1.0, 0.0], k=1)  # Vetor de busca com dimensão errada