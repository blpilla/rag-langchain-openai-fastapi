# langchain-openai-fastapi

Um sistema de Recuperação Aumentada por Geração (RAG) utilizando Python, LangChain, OpenAI e FastAPI.

## Visão Geral

Este projeto implementa um sistema RAG capaz de processar documentos, convertê-los em representações vetoriais, armazená-los eficientemente e fornecer respostas contextualizadas através de uma API REST.

## Requisitos

- Python 3.12.5+
- Dependências listadas em `requirements.txt`

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

2. Use curl para interagir com a API:

   - Processar um documento:
     ```bash
     curl -X POST "http://localhost:8000/process_document" \
          -H "Content-Type: application/json" \
          -d '{"content": "O Python é uma linguagem de programação de alto nível, interpretada de script, imperativa, orientada a objetos, funcional, de tipagem dinâmica e forte. Foi lançada por Guido van Rossum em 1991."}'
     ```

   - Fazer uma consulta:
     ```bash
     curl -X POST "http://localhost:8000/query" \
          -H "Content-Type: application/json" \
          -d '{"question": "Quem criou a linguagem Python?"}'
     ```

3. Acesse a documentação da API em `http://localhost:8000/docs`

## Estrutura do Projeto

```
langchain-openai-fastapi/
│
├── src/
│   ├── document_processor.py
│   ├── vector_db.py
│   └── rag_engine.py
│
├── tests/
│   ├── test_document_processor.py
│   ├── test_vector_db.py
│   └── test_rag_engine.py
│
├── .env
├── .gitignore
├── main.py
├── requirements.txt
└── README.md
```

## Componentes Principais

- `document_processor.py`: Responsável por processar e segmentar documentos.
- `vector_db.py`: Implementa o armazenamento e busca de vetores usando FAISS.
- `rag_engine.py`: Coordena a recuperação de informações e geração de respostas usando LLM.
- `main.py`: Implementa a API FastAPI para interagir com o sistema RAG.

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