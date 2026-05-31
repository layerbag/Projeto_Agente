# LLM Orchestration

Projeto experimental de orquestração de LLMs com **RAG híbrido** (PGVector + FTS), **agente conversacional com tools** (LangGraph), sumarização de documentos, e interfaces via **CLI**, **API FastAPI** e **página web estática**.

O usuário pode indexar PDFs e transcrições de vídeos do YouTube, fazer perguntas sobre o conteúdo, obter resumos e tópicos, e consultar informações atualizadas na web.

## Stack

- Python 3.12+, Poetry
- LangChain, LangGraph
- FastAPI + Uvicorn
- PostgreSQL + pgvector (PGVector)
- ChromaDB (fallback legado)
- HuggingFace Embeddings (`BAAI/bge-m3`)
- CrossEncoder (`BAAI/bge-reranker-v2-m3`)
- rank-bm25
- Groq API (`llama-3.1-8b-instant`), Google Gemini API (`gemini-2.5-flash` — fallback)
- LLMLingua (compressão de contexto)
- YouTube Transcript API, yt-dlp, pypdf

## Estrutura do projeto

```
llm-orchestration/
├── api/                     # API FastAPI + endpoints REST
│   ├── main.py
│   └── schemas.py
├── data/                    # Dados persistentes locais
│   └── chroma_db/
├── src/
│   ├── agent/               # Grafo do agente, tools e prompts
│   ├── chains/              # Exemplos de chains e agentes
│   ├── mlops/               # Logging, tracing, métricas e observabilidade
│   ├── rag/                 # Ingestão, vector store, sumarização e RAG chain
│   └── utils/               # Utilitários de arquivos e documentos
├── tests/
├── web/
│   └── index.html           # Interface web estática
├── .env.example
├── docker-compose.yml       # PostgreSQL + pgvector
├── pyproject.toml
├── poetry.lock
└── README.md
```

## Pré-requisitos

- **Docker** (para o banco PostgreSQL com pgvector)
- **Python 3.12+**
- **Poetry**
- Chaves de API: Groq (obrigatório), Google Gemini (recomendado para fallback)

## Instalação

### 1. Clone e instale dependências

```bash
git clone <url-do-repositorio>
cd llm-orchestration
poetry install
```

### 2. Suba o banco PostgreSQL

```bash
docker compose up -d
```

Isso inicia um container com PostgreSQL 16 + pgvector na porta `5432`, com banco `orchestration_db`.

### 3. Configure as variáveis de ambiente

```bash
cp .env.example .env
```

Edite o `.env` com suas chaves:

```env
GROQ_API_KEY=your_groq_key_here
GOOGLE_API_KEY=your_google_key_here
LANGCHAIN_API_KEY=your_langsmith_key_here
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=llm-orchestration
CHROMA_PERSIST_DIR=./data/chroma_db
EMBEDDING_MODEL=all-MiniLM-L6-v2
```

Apenas `GROQ_API_KEY` é obrigatória; as demais são opcionais.

## Executando pela CLI

```bash
poetry run python -m src.rag.main
```

Menu interativo:
- **I** — indexar documentos (PDF ou YouTube)
- **C** — iniciar chat com RAG
- **sair** — encerrar

### Indexar documentos

```text
Digite "I" para inserir novos documento ou "C" para iniciar o chat: i
Digite "tipo(pdf ou vídeo) caminho" para indexar um documento ou "fim" para concluir:
pdf /caminho/para/arquivo.pdf
video https://www.youtube.com/watch?v=ID_DO_VIDEO
fim
```

### Chat

```text
Digite "I" para inserir novos documento ou "C" para iniciar o chat: c
Digite sua pergunta (ou "sair" para encerrar): Qual o resumo do documento?
```

## Executando a API

```bash
poetry run uvicorn api.main:app --reload
```

Com a API rodando:

| Rota | Descrição |
|---|---|
| `GET /` | Health check |
| `GET /ui` | Interface web estática |
| `GET /docs` | Documentação Swagger |
| `POST /chat` | Envia mensagem para o agente |
| `POST /indexDoc` | Indexa PDF ou YouTube |
| `GET /documents` | Lista documentos indexados |
| `GET /sessions_db` | Lista sessões |
| `GET /sessions/{id}` | Histórico de uma sessão |
| `DELETE /sessions/{id}` | Remove uma sessão |

### Exemplos

```bash
# Chat
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Qual o resumo do documento X?"}'

# Indexar PDF
curl -X POST http://127.0.0.1:8000/indexDoc \
  -F "type=pdf" -F "file=@/caminho/para/arquivo.pdf"

# Indexar YouTube
curl -X POST http://127.0.0.1:8000/indexDoc \
  -F "type=YouTube" -F "url=https://www.youtube.com/watch?v=ID"
```

## Fluxo da aplicação

### Ingestão

```
PDF / YouTube
  → load_pdf() / extrair_transcricao()
  → summarize_document() (Map-Reduce via LLM)
  → salvar_sumario() (tabela document_summaries)
  → semantic_chunking()
  → PGVector (langchain_pg_embedding)
```

### Pergunta do usuário (RAG clássico)

```
Pergunta
  → Contextualização com histórico
  → Busca híbrida: PGVector + FTS + RRF
  → Re-ranking com CrossEncoder
  → Compressão com LLMLingua
  → Resposta via Groq (fallback Gemini)
```

### Agente conversacional (LangGraph)

```
START → agent (LLM decide: responde ou chama tool)
  ├─ tool_calls → tool_node → verifica_final
  │   ├─ {"final": true} → final_response → END
  │   └─ senão → agent → END
  └─ sem tool_calls → END
```

## Persistência local

- **PostgreSQL (`orchestration_db`)**: vetores (`langchain_pg_embedding`), sumários (`document_summaries`)
- **`checkpoints.sqlite`**: checkpoints do LangGraph (memória das conversas)
- **`rag.log`**: logs da aplicação
- **`data/chroma_db/`**: fallback legado do ChromaDB

## Licença

MIT
