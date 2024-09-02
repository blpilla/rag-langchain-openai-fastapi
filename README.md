# langchain-openai-fastapi

Um sistema de Recuperação Aumentada por Geração (RAG) utilizando Python, LangChain, OpenAI e FastAPI.

## Visão Geral

Este projeto implementa um sistema RAG capaz de processar documentos, convertê-los em representações vetoriais, armazená-los eficientemente e fornecer respostas contextualizadas através de uma API REST.

## Requisitos

- Python 3.12.5
- langchain 0.2.14
- openai 0.27.2
- faiss-cpu 1.8.0
- fastapi 0.109.2
- uvicorn 0.27.1
- python-dotenv 1.0.0
- pydantic 2.6.1

## Instalação

1. Clone o repositório:
   ```
   git clone https://github.com/blpilla/langchain-openai-fastapi.git
   cd langchain-openai-fastapi
   ```

2. Crie e ative um ambiente virtual:
   ```
   python -m venv venv
   source venv/bin/activate  # No Windows use `venv\Scripts\activate`
   ```

3. Instale as dependências:
   ```
   pip install -r requirements.txt
   ```

4. Configure as variáveis de ambiente:
   Crie um arquivo `.env` na raiz do projeto e adicione sua chave API do OpenAI:
   ```
   OPENAI_API_KEY=sua_chave_api_aqui
   ```

## Uso

1. Inicie o servidor:
   ```
   uvicorn main:app --reload
   ```

2. Acesse a documentação da API em `http://localhost:8000/docs`

## Estrutura do Projeto

```
langchain-openai-fastapi/
│
├── src/
│   ├── document_processor.py
│   ├── text_to_vector.py
│   ├── vector_db.py
│   └── rag_engine.py
│
├── tests/
│   ├── test_document_processor.py
│   ├── test_text_to_vector.py
│   ├── test_vector_db.py
│   └── test_rag_engine.py
│
├── .env
├── .gitignore
├── main.py
├── requirements.txt
└── README.md
```

## Características Principais

- Processamento e segmentação de documentos
- Conversão de texto para vetores usando OpenAI Embeddings
- Armazenamento eficiente de vetores usando FAISS
- Motor RAG para recuperação de informações e geração de respostas
- API REST com FastAPI para interação com o sistema

## Contribuindo

Contribuições são bem-vindas! Por favor, leia as diretrizes de contribuição antes de submeter pull requests.

## Licença

Este projeto está licenciado sob a [MIT License](https://opensource.org/licenses/MIT).
