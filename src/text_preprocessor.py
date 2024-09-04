import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Baixa os recursos necessários do NLTK
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

class TextPreprocessor:
    def __init__(self, language='portuguese'):
        """
        Inicializa o processador de texto com o idioma especificado.
        
        Parâmetros:
            language (str): O idioma para o qual o processador de texto será inicializado (padrão: 'portuguese').
        
        Não retorna nada.
        """
        # Inicializa o conjunto de stopwords para o idioma especificado
        self.stop_words = set(stopwords.words(language))

    def preprocess(self, text):
        """
        Pré-processa um texto, convertendo-o para minúsculas, tokenizando-o em palavras individuais, removendo as stopwords e reconstruindo o texto a partir dos tokens processados para uma vetorização otimizada e o desenvolvimento do modelo com alta performance.

        Parâmetros:
            text (str): O texto a ser pré-processado.

        Retorna:
            str: O texto pré-processado.
        """
        # Converte o texto para minúsculas
        text = text.lower()
        
        # Tokeniza o texto em palavras individuais
        tokens = word_tokenize(text)
        
        # Remove as stopwords
        tokens = [token for token in tokens if token not in self.stop_words]
        
        # Reconstrói o texto a partir dos tokens processados
        preprocessed_text = ' '.join(tokens)
        
        return preprocessed_text