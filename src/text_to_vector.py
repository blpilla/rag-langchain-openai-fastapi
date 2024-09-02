from langchain_openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()

class TextToVector:
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY não encontrada nas variáveis de ambiente")
        self.embeddings = OpenAIEmbeddings(openai_api_key=api_key)

    def convert(self, text):
        return self.embeddings.embed_query(text)