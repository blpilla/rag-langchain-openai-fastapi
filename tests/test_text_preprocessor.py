import pytest
from src.text_preprocessor import TextPreprocessor

@pytest.fixture
def preprocessor():
    """
    Cria uma instância do preprocessador de texto para uso nos testes.

    Parâmetros:
        None

    Retorna:
        Uma instância do TextPreprocessor configurada para o idioma português.
    """
    # Cria uma instância do preprocessador para uso nos testes
    return TextPreprocessor(language='portuguese')

def test_basic_preprocessing(preprocessor):
    # Testa o pré-processamento básico de um texto simples
    text = "Este é um texto de teste. Ele contém algumas palavras comuns."
    expected = "texto teste . contém algumas palavras comuns ."
    
    result = preprocessor.preprocess(text)
    assert result == expected, f"Expected '{expected}', but got '{result}'"

def test_stopwords_removal(preprocessor):
    # Verifica se as stopwords em português são removidas corretamente
    text = "O a os as um uma uns umas são removidas do texto."
    expected = "uns umas removidas texto ."
    
    result = preprocessor.preprocess(text)
    assert result == expected, f"Expected '{expected}', but got '{result}'"

def test_case_insensitivity(preprocessor):
    # Testa a conversão para minúsculas
    text = "TEXTO EM MAIÚSCULAS e Texto em Minúsculas"
    expected = "texto maiúsculas texto minúsculas"
    
    result = preprocessor.preprocess(text)
    assert result == expected, f"Expected '{expected}', but got '{result}'"

def test_punctuation_preservation(preprocessor):
    # Verifica se a pontuação é preservada após o pré-processamento
    text = "Olá, mundo! Como vai você? Tudo bem: sim."
    expected = "olá , mundo ! vai ? tudo bem : sim ."
    
    result = preprocessor.preprocess(text)
    assert result == expected, f"Expected '{expected}', but got '{result}'"

def test_number_preservation(preprocessor):
    # Testa se os números são mantidos no texto pré-processado
    text = "Tenho 3 maçãs e 5 laranjas."
    expected = "3 maçãs 5 laranjas ."
    
    result = preprocessor.preprocess(text)
    assert result == expected, f"Expected '{expected}', but got '{result}'"

def test_special_characters(preprocessor):
    # Verifica o tratamento de caracteres especiais e e-mails
    text = "E-mail: exemplo@email.com, site: www.exemplo.com.br"
    expected = "e-mail : exemplo @ email.com , site : www.exemplo.com.br"
    
    result = preprocessor.preprocess(text)
    assert result == expected, f"Expected '{expected}', but got '{result}'"

def test_empty_input(preprocessor):
    # Testa o comportamento com entrada vazia
    text = ""
    expected = ""
    
    result = preprocessor.preprocess(text)
    assert result == expected, "Empty input should return empty string"

def test_only_stopwords(preprocessor):
    # Verifica o comportamento quando o texto contém apenas stopwords
    text = "o a um uma"
    expected = ""
    
    result = preprocessor.preprocess(text)
    assert result == expected, f"Expected '{expected}', but got '{result}'"