# `src/agent/` — Agente conversacional

Implementa um **agente com tool-calling** usando LangGraph. O LLM (Groq com fallback Gemini) decide se responde diretamente ou se chama uma ferramenta para buscar informações.

## Arquivos

| Arquivo | Descrição |
|---|---|
| `agent_graph.py` | Grafo LangGraph com nós `agent`, `tools`, `final_response`. Inclui versão async com SSE streaming e versão sync para invoke direto. |
| `tools.py` | 5 tools: `search_knowledge_base`, `web_search`, `list_documents`, `get_document_summary`, `get_document_topics`. As tools de sumário/tópicos verificam cache na tabela `document_summaries` antes de chamar o LLM. |
| `prompts.py` | `SYSTEM_PROMPT` que define o comportamento do agente. |
| `__init__.py` | Exporta `agent_graph` e `tools_list`. |

## Fluxo do grafo

```
START → agent (LLM + tools)
  ├─ LLM responde texto → END
  └─ LLM chama tool → tool_node
      └─ verifica_final
          ├─ {"final": true} → final_response → END
          └─ senão → agent → END
```

O nó `final_response` permite que tools como `get_document_summary` retornem `{"final": true, "result": "..."}` e a resposta vá direto para o usuário sem passar pelo LLM novamente.

## Streaming

`stream_agent_events()` emite eventos SSE:
- `thinking` — LLM começou a processar
- `token` — tokens da resposta (streaming)
- `tool` / `tool_result` — tool iniciou / concluiu
- `done` — resposta final
