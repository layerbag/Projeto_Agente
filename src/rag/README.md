# `src/rag/` — Core RAG

Módulo central do projeto. Gerencia ingestão de documentos, armazenamento vetorial, sumarização, busca híbrida e geração de respostas.

## Arquivos

| Arquivo | Descrição |
|---|---|
| `ingestao.py` | Carrega PDFs (`PyPDFLoader`) e transcrições YouTube (`YouTubeTranscriptApi`). Divide em chunks semânticos por similaridade de cosseno com fallback `RecursiveCharacterTextSplitter`. |
| `vector_store.py` | Configura embeddings `BAAI/bge-m3` (GPU) e gerencia conexão com PostgreSQL via PGVector. Funções: `criar_vector_store()`, `carregar_vector_store()`, `busca_semantica()`, `get_engine()`. |
| `rag_chain.py` | Pipeline RAG completo: busca híbrida (pgvector + FTS + RRF), re-ranking (CrossEncoder), compressão (LLMLingua), contextualização e geração. Compila o grafo LangGraph `app`. |
| `document_summarizer.py` | Sumarização Map-Reduce: divide o texto em chunks, resume cada um (Map) e combina em um resumo geral + tópicos (Reduce). Usa LLM com retry para rate limits. |
| `document_summary_store.py` | CRUD para a tabela `document_summaries` no PostgreSQL. Funções: `criar_tabela()`, `salvar_sumario()`, `buscar_sumario()`, `buscar_topicos_por_titulo()`, `listar_sumarios()`. |
| `main.py` | CLI interativa: menu para indexar documentos e iniciar chat RAG. |
| `logger.py` | Logger simples com `logging.basicConfig`. |

## Fluxo de ingestão

```
PDF / YouTube
  → load_pdf() / extrair_transcricao()
  → summarize_document()  (Map-Reduce)
  → salvar_sumario()      (cache na tabela document_summaries)
  → semantic_chunking()
  → PGVector (langchain_pg_embedding)
```

## Fluxo de busca (RAG)

```
Pergunta
  → contextualize_question() (reescreve com histórico)
  → busca_hibrida_pgvector() (pgvector + FTS + RRF)
  → reranker (CrossEncoder)
  → llmlingua (compressão)
  → llm.invoke() (Groq → Gemini)
  → resposta
```
