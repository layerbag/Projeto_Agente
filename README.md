# LLM Orchestration

Projeto experimental de orquestração de LLMs com **RAG híbrido** (PGVector + FTS), **agente conversacional baseado em grafo** (LangGraph) com **ferramentas modulares**, sumarização Map-Reduce, streaming SSE, e interface **web SPA** + **API FastAPI**.

O usuário pode indexar PDFs e transcrições do YouTube, perguntar sobre o conteúdo indexado, obter resumos, deletar documentos, consultar a web, e inspecionar logs — tudo via agente ou chamadas diretas.

## Stack

- Python 3.12+, Poetry
- LangChain, LangGraph (+ checkpoint SQLite)
- FastAPI + Uvicorn (SSE streaming)
- PostgreSQL 16 + pgvector
- HuggingFace Embeddings (`BAAI/bge-m3`, `sentence-transformers`)
- CrossEncoder (`BAAI/bge-reranker-v2-m3`) — re-ranking
- rank-bm25 — busca lexical
- Groq (`llama-3.1-8b-instant`) + Google Gemini (`gemini-2.5-flash`, fallback)
- LLMLingua — compressão de contexto
- YouTube Transcript API, yt-dlp, pypdf

## Estrutura do projeto

```
llm-orchestration/
├── api/                      # API FastAPI
│   ├── main.py               #   Rotas REST + SSE, CORS, lifespan
│   └── schemas.py            #   Pydantic models (request/response)
├── web/
│   └── index.html            #   SPA: Chat, Documentos, Logs, Retrieval
├── src/
│   ├── agent/                # Agente LangGraph
│   │   ├── agent_graph.py    #   Grafo (agent, tools, final/confirmation nodes)
│   │   ├── skills/           #   Skill Registry pattern
│   │   │   ├── registry.py   #     AgentSkill ABC + SkillRegistry
│   │   │   ├── rag_skill.py  #     search, list, index, get_title, delete
│   │   │   ├── web_skill.py  #     web_search (DuckDuckGo)
│   │   │   ├── document_skill.py  # summarize, extract bullets
│   │   │   └── logs_skill.py #     get_logs (rag.log)
│   │   └── __init__.py       #   Exporta agent_graph + registry
│   ├── rag/                  # RAG engine
│   │   ├── rag_chain.py      #   LLM setup (Groq+Gemini), hybrid search (RRF)
│   │   ├── vector_store.py   #   PGVector + embeddings (BGE-M3)
│   │   ├── ingestao.py       #   Chunking semântico, YouTube extraction
│   │   ├── document_summarizer.py  # Map-Reduce summarization
│   │   └── document_summary_store.py  # Cache de sumários (PostgreSQL)
│   ├── utils/
│   │   └── project_utils.py  #   PDF loading, caminho, delete_indexed_document
│   └── mlops/                # Observabilidade
│       ├── logging.py        #   Logger (console + rag.log)
│       ├── metrics.py        #   @measure_time decorator
│       ├── tracing.py        #   trace_retrieval (debug detalhado)
│       ├── evaluation.py     #   evaluate_response (qualidade básica)
│       └── observability.py  #   Facade (observe_query/docs/response)
├── data/
│   └── chroma_db/            # Legado (não usado ativamente)
├── tests/
├── docker-compose.yml        # PostgreSQL 16 + pgvector
├── .env.example
├── pyproject.toml
├── poetry.lock
└── README.md
```

## Pré-requisitos

- **Docker** (banco PostgreSQL com pgvector)
- **Python 3.12+**
- **Poetry**
- Chaves de API: Groq (obrigatório), Google Gemini (recomendado para fallback)

## Instalação

```bash
git clone <url-do-repositorio>
cd llm-orchestration
poetry install
```

Suba o banco:

```bash
docker compose up -d
```

Container: PostgreSQL 16 + pgvector, porta `5432`, database `orchestration_db`.

Configure o `.env`:

```bash
cp .env.example .env
```

```env
GROQ_API_KEY=your_groq_key_here
GOOGLE_API_KEY=your_google_key_here
LANGCHAIN_API_KEY=your_langsmith_key_here
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=llm-orchestration
```

Apenas `GROQ_API_KEY` é obrigatória.

## Executando a API

```bash
poetry run uvicorn api.main:app --reload
```

### Rotas da API

| Método | Rota | Descrição |
|--------|------|-----------|
| GET | `/` | Serve `web/index.html` (SPA) |
| POST | `/agent/chat/stream` | Chat com agente — streaming SSE |
| POST | `/agent/chat` | Chat com agente — síncrono |
| POST | `/chat` | Chat RAG legado (desativado) |
| POST | `/indexDoc` | Indexar PDF (upload) ou YouTube (URL) |
| GET | `/indexDoc/status/{task_id}` | Polling de task de indexação |
| GET | `/documents` | Listar documentos indexados |
| DELETE | `/documents` | Deletar documento por título |
| POST | `/retrieval/debug` | Debug da busca híbrida (semântico + BM25 + rerank) |
| GET | `/sessions_db` | Listar sessões do checkpointer |
| GET | `/sessions/{session_id}` | Histórico de uma sessão |
| DELETE | `/sessions/{session_id}` | Remover sessão |

### Exemplos

```bash
# Chat streaming (SSE)
curl -X POST http://127.0.0.1:8000/agent/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"message": "liste os documentos"}'

# Chat síncrono
curl -X POST http://127.0.0.1:8000/agent/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "o que diz o documento X?"}'

# Indexar PDF
curl -X POST http://127.0.0.1:8000/indexDoc \
  -F "type=pdf" -F "file=@/caminho/para/arquivo.pdf"

# Indexar YouTube
curl -X POST http://127.0.0.1:8000/indexDoc \
  -F "type=YouTube" -F "url=https://www.youtube.com/watch?v=ID"
```

## Agente Conversacional (LangGraph)

```
START → agent (LLM decide: responde ou chama tool)
  ├─ sem tool_calls → END
  └─ tool_calls → tools → route_after_tools
       ├─ {"final": true} → final_response → END
       ├─ {"requires_confirmation": true} → confirmation_response → END
       └─ senão → agent (loop)
```

- **agent**: LLM (Groq com fallback Gemini) + tools bindings
- **tools**: ToolNode executa as tools chamadas pelo LLM
- **route_after_tools**: Roteia baseado nas flags retornadas pela tool
  - `final: true` → exibe resultado direto, sem LLM
  - `requires_confirmation: true` → pede confirmação ao usuário
  - caso contrário → volta pro agente processar o resultado
- **final_response**: Extrai `result` do JSON e devolve como resposta final
- **confirmation_response**: Pergunta ao usuário se deseja prosseguir

### Tools disponíveis (4 skills)

| Skill | Tools | Descrição |
|-------|-------|-----------|
| **rag** | `search_knowledge_base` | Busca híbrida na base local |
| | `list_documents` | Lista documentos indexados |
| | `index_document` | Indexa PDF ou YouTube |
| | `get_title` | Busca título exato (pré-deleção) |
| | `delete_document` | Deleta documento (requer confirmação) |
| **web** | `web_search` | Busca DuckDuckGo |
| **document** | `summarize_document` | Resumo completo (com cache) |
| | `extract_document_bullets` | Tópicos em bullet points |
| **logs** | `get_logs` | Consulta `rag.log` (filtro por nível/palavra) |

As skills se registram automaticamente via `SkillRegistry` (pattern Registry) ao importar `src.agent.skills`.

## Pipeline de ingestão

```
PDF / YouTube
  → load_pdf() / extrair_transcricao()
  → dividir_em_chunks() (chunking semântico por similaridade coseno)
  → add_documents() → PGVector (langchain_pg_embedding)
  → summarize_document() (Map-Reduce via LLM)
  → salvar_sumario() (tabela document_summaries)
```

## Busca híbrida (query)

```
Pergunta
  → Busca semântica (pgvector <=>, top 20)
  → Busca textual (PostgreSQL FTS ts_rank_cd, top 20)
  → RRF (Reciprocal Rank Fusion, k=60)
  → CrossEncoder (re-ranking, configurável)
  → LLMLingua (compressão de contexto)
  → Resposta via Groq (fallback Gemini)
```

A busca híbrida é feita em **uma única query SQL** que combina os dois rankings via RRF.

## Observabilidade (MLOps)

- **`@measure_time(name)`**: decorator que loga duração de qualquer função/tool
- **`observe_query`**: loga a pergunta do usuário
- **`observe_docs`**: loga quantidade de documentos recuperados + trace detalhado
- **`observe_response`**: loga tamanho da resposta + checagem de qualidade (evaluate_response)
- **`trace_retrieval`**: log completo (query original, rewrited query, cada doc com score e preview)
- **`evaluate_response`**: alerta se resposta é muito curta ou contém padrões de alucinação

## Persistência

- **PostgreSQL (`orchestration_db`)**: vetores (`langchain_pg_embedding`), sumários (`document_summaries`)
- **`checkpoints.sqlite`**: checkpoints do LangGraph (memória de conversas por `thread_id`)
- **`rag.log`**: logs da aplicação

## Licença

MIT
