# `src/utils/` — Utilitários

Funções auxiliares compartilhadas entre os módulos do projeto.

## Arquivos

| Arquivo | Descrição |
|---|---|
| `project_utils.py` | `normalizar_caminho()` — normaliza paths (Linux/WSL/Windows). `load_pdf()` — carrega PDF com `PyPDFLoader` e enriquece metadados. `yt2doc()` — converte transcript + metadata em `Document`. `delete_indexed_document()` — remove registros do PGVector por título. |
| `test_setup.py` | Script de validação do ambiente: testa conectividade com Groq, Gemini, embeddings e ChromaDB. |
