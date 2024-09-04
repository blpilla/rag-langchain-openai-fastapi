# Minhas Decisões Técnicas

Aqui estão as principais decisões que tomei ao desenvolver este sistema RAG e por que as fiz:

## 1. Escolhi o FAISS como Banco de Dados Vetorial

Optei pelo FAISS (Facebook AI Similarity Search) porque:

- É super rápido para buscar vetores semelhantes, mesmo em conjuntos enormes de dados.
- Pode lidar com bilhões de vetores, então o sistema pode crescer sem problemas.
- Oferece várias opções de indexação, permitindo ajustar entre velocidade e precisão.
- Tem boa integração com o Python.

## 2. Adicionei um Pré-processador de Linguagem Natural

Decidi incluir uma etapa de pré-processamento de linguagem natural porque:

- Melhora a qualidade dos vetores, o que leva a buscas mais precisas.
- Permite limpar e padronizar o texto antes de converter em vetores.
- Possibilita ajustar facilmente o tratamento do texto para diferentes idiomas ou tipos de documento.
- Reduz a quantidade de dados e otimiza o armazenamento e processamento, melhorando a performance em larga escala.

## 3. Adotei uma Estratégia de Persistência

Implementei a persistência do banco de dados vetorial porque:

- Economiza tempo e recursos ao não precisar reprocessar todos os documentos a cada reinicialização.
- Permite que o sistema mantenha seu conhecimento entre as sessões.
- Facilita o backup e a migração dos dados, se necessário.

## 4. Escolhi "stuff" como Chain Type Padrão

Optei por usar "stuff" como o tipo de chain padrão no RAGEngine porque:

- Proporciona respostas mais coerentes quando o número de documentos relevantes é limitado.
- Mantém o contexto completo em uma única passagem pelo modelo, ideal para consultas que requerem compreensão holística.
- É mais eficiente em termos de chamadas à API, reduzindo custos para conjuntos de dados menores.

Obs: Para una base de dados em larga escala, a escolha do tipo de chain seria: "refine", ou "map reduce".

Estas decisões foram tomadas pensando em criar um sistema RAG capaz de processar e recuperar informações de forma precisa e escalável.