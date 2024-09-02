import pytest
from src.text_to_vector import TextToVector
import os

@pytest.fixture
def text_to_vector():
    return TextToVector()

def test_convert_text_to_vector(text_to_vector):
    text = "Este é um texto de teste."
    vector = text_to_vector.convert(text)
    
    assert isinstance(vector, list), "O resultado deve ser uma lista"
    assert len(vector) == 1536, "O vetor deve ter 1536 dimensões (padrão para OpenAI)"
    assert all(isinstance(x, float) for x in vector), "Todos os elementos do vetor devem ser floats"

def test_empty_text(text_to_vector):
    vector = text_to_vector.convert("")
    assert len(vector) == 1536, "Mesmo para texto vazio, deve retornar um vetor de 1536 dimensões"

def test_different_texts_produce_different_vectors(text_to_vector):
    text1 = "Este é o primeiro texto."
    text2 = "Este é um texto diferente."
    
    vector1 = text_to_vector.convert(text1)
    vector2 = text_to_vector.convert(text2)
    
    assert vector1 != vector2, "Textos diferentes devem produzir vetores diferentes"

def test_missing_api_key():
    # Temporarily remove the API key
    original_key = os.environ.get("OPENAI_API_KEY")
    os.environ.pop("OPENAI_API_KEY", None)
    
    with pytest.raises(ValueError, match="OPENAI_API_KEY não encontrada nas variáveis de ambiente"):
        TextToVector()
    
    # Restore the API key
    if original_key:
        os.environ["OPENAI_API_KEY"] = original_key