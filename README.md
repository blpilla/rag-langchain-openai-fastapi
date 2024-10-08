# Sistema RAG com Python, LangChain, OpenAI, Faiss, NLTK e FastAPI

Este projeto implementa um sistema de Geração Aumentada de Recuperação (RAG) utilizando o seguinte conjunto de tecnologias:

- **Python** como linguagem base
- **LangChain** para orquestração de componentes de IA
- **OpenAI** para geração de embeddings e processamento de linguagem natural
- **Faiss** (Facebook AI Similarity Search) para armazenamento e busca de vetores
- **NLTK** (Natural Language Toolkit) para pré-processamento de texto
- **FastAPI** para interface de API de alta performance
- Bibliotecas especializadas para processamento de documentos: **PyPDF2**, **python-docx**, **openpyxl**, **BeautifulSoup4**

## Visão Geral

O sistema RAG desenvolvido é capaz de processar documentos em diversos formatos (incluindo TXT, PDF, DOCX, XLSX, HTML, CSV e outros formatos de texto), convertê-los em representações vetoriais utilizando modelos de embedding, pré-processá-los em linguagem natural, persistir o conhecimento aprendido e fornecer respostas contextualizadas através de uma API REST.

![Diagrama do Sistema RAG](./images/simplified_technical_drawing.svg)

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

2. Acesse a documentação interativa da API:
   Abra um navegador e vá para `http://localhost:8000/docs`

3. Teste a API usando o Swagger UI:
   
   a. Upload de Documentos:
      - Clique no endpoint POST `/upload_documents`
      - Clique em "Try it out"
      - Use o botão "Choose File" para selecionar um ou mais arquivos
      - Clique em "Execute" para fazer o upload

   b. Fazer uma Consulta:
      - Clique no endpoint POST `/query`
      - Clique em "Try it out"
      - Insira sua pergunta no campo "question" do corpo da requisição
      - Clique em "Execute" para enviar a consulta

4. Alternativamente, use curl para interagir com a API:

   Upload de documentos:
   ```
   curl -X POST "http://localhost:8000/upload_documents" \
        -H "Content-Type: multipart/form-data" \
        -F "files=@/caminho/para/seu/arquivo.pdf" \
        -F "files=@/caminho/para/seu/arquivo.csv"
   ```

   Fazer uma consulta:
   ```
   curl -X POST "http://localhost:8000/query" \
        -H "Content-Type: application/json" \
        -d '{"question": "Qual é o tema principal dos documentos?"}'
   ```

## Executando Testes Unitários

Para executar os testes do projeto, siga estas etapas:

1. Certifique-se de que está no diretório raiz do projeto.

2. Configure o PYTHONPATH para incluir o diretório atual:
   ```
   export PYTHONPATH=./
   ```

3. Execute os testes usando pytest:
   ```
   pytest tests/ -vv
   ```

   Este comando executará todos os testes no diretório `tests/` com saída detalhada (-vv).


## Estrutura do Projeto

```

├── src/
│   ├── document_processor.py
│   ├── text_preprocessor.py
│   ├── vector_db.py
|   └── rag_engine.py
├── tests/
│   ├── test_document_processor.py
│   ├── test_text_preprocessor.py
│   ├── test_main.py
│   ├── test_vector_db.py
│   └── test_rag_engine.py
├── persistent_vector_db/
│   ├── index.faiss
│   └── index.pkl
├── .env
├── .gitignore
├── main.py
├── requirements.txt
├── ARCHITECTURE.md
└── README.md

```

## Características Principais

- Processamento de múltiplos formatos de documento (TXT, PDF, DOCX, XLSX, HTML, CSV e outros formatos de texto)
- Pré-processamento de texto para melhorar a qualidade dos vetores e otimizar o desempenho
- Conversão de texto para vetores usando OpenAI Embeddings
- Armazenamento eficiente de vetores usando FAISS
- Motor RAG para recuperação de informações e geração de respostas
- API REST com FastAPI para interação com o sistema
- Documentação interativa com Swagger UI
- Persistência do banco de dados vetorial para manter o conhecimento

## Contribuindo

Contribuições são bem-vindas! Por favor, leia as diretrizes de contribuição antes de submeter pull requests.

## Licença

Este projeto está licenciado sob a [MIT License](https://opensource.org/licenses/MIT).