# `src/chains/` — Exemplos de chains

Módulo com implementações de referência de padrões LangChain. **Não é usado em produção** — serve como material de estudo e testes rápidos.

## Arquivos

| Arquivo | Descrição |
|---|---|
| `basic_chain.py` | Chain minimalista: prompt → Groq → `StrOutputParser` com streaming. |
| `paralel_chains.py` | Demonstra `RunnableParallel`: executa 3 chains em paralelo (prós, contras, sumário) para um tópico. |
| `chat_with_memory.py` | Exemplo de `RunnableWithMessageHistory` com `ChatMessageHistory` em memória. |
