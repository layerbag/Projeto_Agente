# pyrefly: ignore [missing-import]
from src.mlops.logging import logger
# pyrefly: ignore [missing-import]
from src.mlops.tracing import trace_retrieval
# pyrefly: ignore [missing-import]
from src.mlops.evaluation import evaluate_response

def observe_query(query):
    logger.info(f"Query: {query}")

def observe_docs(query, rewritten_query, docs):
    logger.info(
        f"Retrieved {len(docs)} docs"
    )

    trace_retrieval(query, rewritten_query, docs)

def observe_response(response):
    logger.info(
        f"Response length: {len(response)}"
    )

    evaluate_response(response)
