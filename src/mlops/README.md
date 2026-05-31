# `src/mlops/` — Observabilidade e operações

Camada de **MLOps** do projeto: logging centralizado, medição de tempo, tracing de retrieved documents, avaliação de respostas e facade de observabilidade.

## Arquivos

| Arquivo | Descrição |
|---|---|
| `logging.py` | Configura logger `rag` com output para console + `rag.log`. |
| `metrics.py` | Decorador `@measure_time(name)` que loga duração de funções sync e async. |
| `tracing.py` | `trace_retrieval()` — loga query original, reescrita e preview dos documentos recuperados. |
| `evaluation.py` | `evaluate_response()` — validações básicas: tamanho mínimo, frases de "não sei". |
| `observability.py` | Facade que reúne tracing + evaluation. Exporta `observe_query()`, `observe_docs()`, `observe_response()`. |

## Uso

```python
from src.mlops.observability import observe_query, observe_docs, observe_response
from src.mlops.metrics import measure_time

@measure_time("my_function")
def my_function():
    observe_query(user_input)
    ...
    observe_docs(query, rewritten, docs)
    ...
    observe_response(response)
```
