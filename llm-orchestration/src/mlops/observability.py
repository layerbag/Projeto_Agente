from langsmith import Client
from langsmith.run_helpers import traceable
import time

client = Client()

@traceable(name="rag_query")
def rag_com_rastreamento(chain, pergunta: str, config: dict) -> dict[str, float]: # type: ignore
    """Executa uma consulta RAG com rastreamento usando LangSmith."""
    start_time = time.time()
    if pergunta.lower().__contains__("documento") or pergunta.lower().__contains__("pdf"):
        base_input = {"query": pergunta, "filter": {"type": "pdf"}}
    elif pergunta.lower().__contains__("vídeo") or pergunta.lower().__contains__("video"):
        base_input = {"query": pergunta, "filter": {"type": "YouTube"}}
    else:
        base_input = {"query": pergunta}

    resposta = chain.invoke(base_input, config={"configurable": {"session_id": "sessao-123"}})  # type: ignore

    latencia = time.time() - start_time

    return {"resposta": resposta, "tempo_gasto": latencia}

def registrar_feedback(run_id: str, score: int, comentario: str = ""):
    """Registra avaliação humana de uma resposta (0.0 a 1.0)"""
    client.create_feedback(
        run_id=run_id,
        key = "qualidade",
        score = score,
        comment = comentario
    )

    print(f"Feedback registrado: {score}/1.0")