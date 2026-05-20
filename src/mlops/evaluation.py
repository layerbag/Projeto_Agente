from src.mlops.logging import logger

def evaluate_response(response: str):

    if len(response.strip()) < 10:
        logger.warning(
            "Very Short Response"
        )
    
    hallucination_patterns = [
        "não sei",
        "não encontrei",
        "não tenho acesso"
    ]

    for pattern in hallucination_patterns:
        if pattern in response.lower():
            logger.warning(
                f"Possible retrieval failure: {pattern}"
            )