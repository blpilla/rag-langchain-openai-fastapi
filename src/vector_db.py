from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()

class VectorDB:
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY não encontrada nas variáveis de ambiente")
        self.embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        self.vector_store = None

    def add(self, documents):
        if self.vector_store is None:
            self.vector_store = FAISS.from_texts(documents, self.embeddings)
        else:
            self.vector_store.add_texts(documents)

    def search(self, query, k=5):
        if self.vector_store is None:
            return []
        return self.vector_store.similarity_search(query, k=k)

    def get_vector_store(self):
        return self.vector_store