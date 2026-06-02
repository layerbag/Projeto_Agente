from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.rag.rag_chain import llm
from src.mlops.logging import logger
import json
import time
import re

MAX_CHARS = 6000
MAX_RETRIES = 3


def call_llm(prompt: str) -> str:
    for attempt in range(MAX_RETRIES):
        try:
            return str(llm.invoke(prompt).content)
        
        except Exception as e:
            error = str(e)
            print(error)
            if "429" in error:
                logger.warning(f"LLM atingiu rate limit tentando novamente em {2 ** attempt}s")
                time.sleep(2 ** attempt)
                continue

            raise
    raise RuntimeError("LLM failed")

def split_document(text: str) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=MAX_CHARS,
        chunk_overlap=200,
        separators=["\n\n","\n",". "," ",""]
    )

    return splitter.split_text(text)

# ========== MAP ==============================

MAP_PROMPT = """
Você receberá um trecho de um documento.
Resuma o conteúdo em no máximo 5 frases.

Trecho:
{chunk}

Retorne somente o resumo.
"""

def summarize_chunk(chunk: str) -> str:

    prompt = MAP_PROMPT.format(
        chunk=chunk
    )

    return call_llm(prompt).strip()

def map_phase(chunks: list[str]) -> list[str]:
    summaries = []
    big_document = len(chunks) >= 50

    for i, chunk in enumerate(chunks):
        if i % 2 != 0 and big_document:
            continue

        print(f"Chunk {i+1}/{len(chunks)}",end='\r',flush=True)

        summary = summarize_chunk(chunk)

        summaries.append(summary)
    
    return summaries

# ======== REDUCE ==========================

REDUCE_PROMPT = """
Você receberá vários resumos de partes de um documento.

Resumos:

{summaries}

Gere:

1. Um resumo geral do documento
2. Uma lista dos principais tópicos

Retorne JSON válido:

{{
    "overall_summary": "...",
    "topics": [
        "...",
        "..."
    ]
}}
"""

def parse_json(text: str):
    start = text.find("{")
    end = text.rfind("}")

    if start == -1:
        return {}
    
    return json.loads(
        text[start:end+1]
    )

def reduce_phase(summaries: list[str]) -> dict:
    
    context = "\n\n".join(summaries)
    
    prompt = REDUCE_PROMPT.format(summaries=context)
    
    response = call_llm(prompt)

    return parse_json(response)

def summarize_document(
    text: str,
    title: str
) -> dict:

    if not text.strip():

        return {
            "title": title,
            "overall_summary": "",
            "topics": []
        }

    chunks = split_document(text)

    logger.info(f"{'='*50}iniciando map_phase{'='*50}")
    summaries = map_phase(chunks)
    logger.info(f"{'='*50}iniciando reduce{'='*50}")
    result = reduce_phase(
        summaries
    )

    return {
        "title": title,
        "overall_summary": result.get(
            "overall_summary",
            ""
        ),
        "topics": result.get(
            "topics",
            []
        )
    }
