# pyrefly: ignore [missing-import]
from src.mlops.logging import logger

def trace_retrieval(query, rewritten_query, docs):

    logger.info("\n\n" + "="*50)
    logger.info(f"QUERY: {query}\nREWRITTEN_QUERY: {rewritten_query}")

    for i, doc in enumerate(docs):

        source = doc.metadata.get(
            "source",
            "unknown"
        )

        tipo = doc.metadata.get(
            "type",
            "unknown"
        )

        titulo = doc.metadata.get(
            "title",
            "unknown"
        )

        rrf_score = doc.metadata.get(
            "rrf_score",
            "unknown"
        )

        preview = doc.page_content[:200]

        logger.info(
            f"[DOC {i+1}] "
            f"type={tipo} "
            f"título={titulo}"
            f"source={source}"
            f"rrf_score={rrf_score}"
        )

        logger.info(preview)

    logger.info("=" * 50 + "\n")
