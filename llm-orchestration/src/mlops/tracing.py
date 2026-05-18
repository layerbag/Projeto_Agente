from src.mlops.logging import logger

def trace_retrieval(query, docs):

    logger.info("="*50)
    logger.info(f"QUERY: {query}")

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

        preview = doc.page_content[:200]

        logger.info(
            f"[DOC {i+1}] "
            f"type={tipo} "
            f"título={titulo}"
            f"source={source}"
        )

        logger.info(preview)

    logger.info("=" * 50)