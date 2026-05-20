# LLM Orchestration

Projeto experimental de orquestração de LLMs com RAG híbrido, memória conversacional e interface via CLI, API FastAPI e página web estática.

O objetivo do projeto é permitir que o usuário indexe documentos PDF e transcrições de vídeos do YouTube em um banco vetorial local, faça perguntas sobre esse conteúdo e receba respostas fundamentadas no contexto recuperado.

## O que o projeto faz

- Indexa PDFs enviados por caminho local ou upload pela API.
- Indexa vídeos do YouTube a partir da transcrição.
- Persiste embeddings localmente com ChromaDB.
- Combina busca semântica, BM25 e re-ranking com CrossEncoder.
- Comprime o contexto recuperado antes de enviar ao modelo.
- Mantém histórico de conversa com LangGraph e SQLite.
- Expõe uma CLI, uma API HTTP e uma interface web simples.

## Como funciona

O fluxo principal fica em `src/rag/rag_chain.py` e é orquestrado com LangGraph.

```text
Pergunta do usuário
        |
        v
Contextualização da pergunta com histórico
        |
        v
Busca semântica no ChromaDB + busca lexical BM25
        |
        v
Mesclagem e remoção de duplicados
        |
        v
Re-ranking com CrossEncoder
        |
        v
Compressão de contexto com LLMChainExtractor
        |
        v
Resposta final usando Groq com fallback para Gemini
```

Durante a ingestão, os documentos são carregados, divididos em chunks e gravados no ChromaDB. O caminho padrão do banco vetorial é `./data/chroma_db`, configurável por variável de ambiente.

## Stack

- Python 3.12+
- Poetry
- LangChain
- LangGraph
- FastAPI
- ChromaDB
- HuggingFace Embeddings
- Sentence Transformers
- CrossEncoder
- rank-bm25
- Groq API
- Google Gemini API
- YouTube Transcript API
- yt-dlp
- pypdf

## Estrutura do projeto

```text
llm-orchestration/
├── api/
│   ├── main.py              # API FastAPI e rotas da interface web
│   └── schemas.py           # Schemas Pydantic
├── data/
│   └── chroma_db/           # Banco vetorial local do ChromaDB
├── src/
│   ├── chains/              # Exemplos de chains e agentes
│   ├── mlops/               # Logging, tracing, métricas e observabilidade
│   ├── rag/                 # Ingestão, vector store e cadeia RAG principal
│   └── utils/               # Utilitários de arquivos e documentos
├── tests/                   # Estrutura de testes
├── web/
│   └── index.html           # Interface web estática
├── .env.example             # Exemplo de variáveis de ambiente
├── pyproject.toml           # Dependências e configuração Poetry
├── poetry.lock              # Lockfile das dependências
└── README.md
```

## Instalação a partir do repositório

Clone o repositório:

```bash
git clone <url-do-repositorio>
cd llm-orchestration
```

Instale as dependências com Poetry:

```bash
poetry install
```

Crie o arquivo `.env` a partir do exemplo:

```bash
cp .env.example .env
```

Edite o `.env` com as suas chaves:

```env
GROQ_API_KEY=your_groq_key_here
GOOGLE_API_KEY=your_google_key_here
LANGCHAIN_API_KEY=your_langsmith_key_here
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=llm-orchestration
CHROMA_PERSIST_DIR=./data/chroma_db
EMBEDDING_MODEL=all-MiniLM-L6-v2
```

As chaves `GROQ_API_KEY` e `GOOGLE_API_KEY` são usadas pelos modelos de resposta e fallback. As variáveis do LangChain/LangSmith são opcionais para tracing, mas já estão previstas no projeto.

## Executando pela CLI

Inicie a aplicação no terminal:

```bash
poetry run python -m src.rag.main
```

Ao abrir, escolha uma operação:

```text
Digite "I" para inserir novos documento ou "C" para iniciar o chat:
```

Para indexar documentos, use:

```text
pdf /caminho/para/arquivo.pdf
video https://www.youtube.com/watch?v=ID_DO_VIDEO
fim
```

Depois, escolha `C` e faça perguntas sobre o conteúdo indexado. Para encerrar o chat, digite `sair`.

## Executando a API e a interface web

Suba a API FastAPI:

```bash
poetry run uvicorn api.main:app --reload
```

Com a API rodando, acesse:

- Health check: `http://127.0.0.1:8000/`
- Interface web: `http://127.0.0.1:8000/ui`
- Documentação Swagger: `http://127.0.0.1:8000/docs`

Rotas principais:

```text
GET    /                     # verifica se a API está rodando
GET    /ui                   # abre a interface web
POST   /chat                 # envia mensagem para o agente
POST   /indexDoc             # indexa PDF ou YouTube
GET    /documents            # lista documentos indexados
GET    /sessions_db          # lista sessões persistidas
GET    /sessions/{session_id} # recupera histórico de uma sessão
DELETE /sessions/{session_id} # remove uma sessão
```

Exemplo de chamada para o chat:

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "O que os documentos dizem sobre o tema principal?"}'
```

Exemplo para indexar um PDF pela API:

```bash
curl -X POST http://127.0.0.1:8000/indexDoc \
  -F "type=pdf" \
  -F "file=@/caminho/para/arquivo.pdf"
```

Exemplo para indexar um vídeo do YouTube:

```bash
curl -X POST http://127.0.0.1:8000/indexDoc \
  -F "type=YouTube" \
  -F "url=https://www.youtube.com/watch?v=ID_DO_VIDEO"
```

## Persistência local

O projeto grava dados locais em:

- `data/chroma_db/`: coleção do ChromaDB com os chunks e embeddings.
- `checkpoints.sqlite`: checkpoints do LangGraph usados para memória das conversas.
- `rag.log`: logs da aplicação RAG.

Esses arquivos permitem continuar usando documentos e históricos já indexados entre execuções.

## Módulos importantes

- `src/rag/ingestao.py`: extrai informações de PDFs e YouTube, cria documentos e chunks.
- `src/rag/vector_store.py`: configura embeddings e acesso ao ChromaDB.
- `src/rag/rag_chain.py`: define o grafo RAG, recuperação, re-ranking, compressão e resposta.
- `src/rag/main.py`: interface CLI para indexação e chat.
- `api/main.py`: API FastAPI, sessão de chat, indexação, listagem de documentos e UI.
- `src/mlops/`: utilitários de métricas, logging, tracing, avaliação e observabilidade.

## Observações de uso

- Na primeira execução, os modelos de embedding e re-ranking podem ser baixados automaticamente.
- A indexação de YouTube depende da disponibilidade de transcrição no vídeo.
- O RAG responde com base no contexto recuperado dos documentos indexados.
- O modelo principal configurado no código é `llama-3.1-8b-instant` via Groq, com fallback para `gemini-2.5-flash`.

## Licença

MIT
