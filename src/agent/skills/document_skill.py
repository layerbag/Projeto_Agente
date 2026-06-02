import time
from langchain.tools import tool
from sqlalchemy import text
from src.rag.rag_chain import busca_hibrida_pgvector, llm
from src.rag.vector_store import get_engine
from src.mlops.metrics import measure_time
from src.mlops.observability import logger
from src.rag.document_summary_store import buscar_topicos_por_titulo, buscar_sumario_por_titulo
from src.agent.skills.registry import AgentSkill, registry


@tool
@measure_time("summarize_document")
def summarize_document(source_query: str) -> dict:
    """APENAS para quando o usuário EXPLICITAMENTE pedir "resumo" ou "sumário" de um documento específico (pelo nome/título).
    NÃO use para responder perguntas sobre o conteúdo do documento — para isso use search_knowledge_base.
    Gera um resumo em parágrafos do documento inteiro."""

    resultados = busca_hibrida_pgvector(source_query, limit=1)

    if not resultados:
        return {
            "result": f"Documento com título/source '{source_query}' não encontrado",
            "final": True
        }

    source = resultados[0].metadata["title"]

    cached = buscar_sumario_por_titulo(source)
    if cached and cached["summary"]:
        return {
            "result": cached["summary"],
            "final": True
        }

    with get_engine().connect() as conn:
        rows = conn.execute(text(r"""
            SELECT document, cmetadata
            FROM langchain_pg_embedding
            WHERE cmetadata->>'title' ILIKE :source
                OR cmetadata->>'source' ILIKE :source
            ORDER BY (cmetadata->>'chunk_index')\:\:int NULLS LAST
        """),
        {"source": f"%{source}%"}
        ).fetchall()

    if not rows:
        return {
            "result": f"Documento com título/source '{source_query}' não encontrado",
            "final": True
        }

    full_text = "\n\n".join(row[0] for row in rows)
    metadata = rows[0][1]
    title = metadata.get("title", source)

    logger.info(f"summarize_document: {title}, {len(full_text)} caracteres, {len(rows)} chunks")

    MAX_CHARS = 50000
    if len(full_text) <= MAX_CHARS:
        return {"result": _summarize(full_text, title), "final": True}

    chunks = [full_text[i:i+MAX_CHARS] for i in range(0, len(full_text), MAX_CHARS)]
    summaries = [
        _summarize(c, f"{title} (parte {i+1}/{len(chunks)})")
        for i, c in enumerate(chunks)
    ]
    final = _summarize("\n\n".join(summaries), title)
    return {"result": final, "final": True}


def _summarize(text: str, title: str) -> str:
    prompt = f""" Resuma o documento abaixo de forma clara e concisa.

    Título: {title}

    Texto:
    {text[:10000]}

    Resumo em português, 3-5 parágrafos:
    """

    max_retries = 5
    for attempt in range(max_retries):
        try:
            return llm.invoke(prompt).content
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "413" in error_str or "rate_limit" in error_str.lower() or "rate limit" in error_str.lower():
                wait = 2 ** attempt
                logger.warning(f"rate limit atingido em '{title}', tentativa {attempt+1}/{max_retries}, aguardando {wait}s...")
                time.sleep(wait)
            else:
                raise
    raise Exception(f"Falha após {max_retries} tentativas por rate limit: '{title}'")


@tool
@measure_time("extract_document_bullets")
def extract_document_bullets(source_query: str) -> dict:
    """APENAS para quando o usuário EXPLICITAMENTE pedir "tópicos" ou "bullet points" de um documento específico (pelo nome/título).
    NÃO use para responder perguntas sobre o conteúdo do documento — para isso use search_knowledge_base.
    Gera uma lista de tópicos em bullet points do documento inteiro."""

    resultados = busca_hibrida_pgvector(source_query, limit=1)

    if not resultados:
        return {
            "result": f"Documento com título/source '{source_query}' não encontrado",
            "final": True
        }

    source = resultados[0].metadata["title"]

    cached = buscar_topicos_por_titulo(source)
    if cached:
        return {
            "result": cached,
            "final": True
        }

    with get_engine().connect() as conn:
        rows = conn.execute(text(r"""
            SELECT document, cmetadata
            FROM langchain_pg_embedding
            WHERE cmetadata->>'title' ILIKE :source
                OR cmetadata->>'source' ILIKE :source
            ORDER BY (cmetadata->>'chunk_index')\:\:int NULLS LAST
        """),
        {"source": f"%{source}%"}
        ).fetchall()

    if not rows:
        return {
            "result": f"Documento com título/source '{source_query}' não encontrado",
            "final": True
        }

    full_text = "\n\n".join(row[0] for row in rows)
    metadata = rows[0][1]
    title = metadata.get("title", source)

    logger.info(f"extract_document_bullets: {title}, {len(full_text)} caracteres, {len(rows)} chunks")

    MAX_CHARS = 50000
    if len(full_text) <= MAX_CHARS:
        return {"result": _topics(full_text, title), "final": True}

    chunks = [full_text[i:i+MAX_CHARS] for i in range(0, len(full_text), MAX_CHARS)]
    topics = [
        _topics(c, f"{title} (parte {i+1}/{len(chunks)})")
        for i, c in enumerate(chunks)
    ]
    return {"result": "\n\n".join(topics), "final": True}


def _topics(text: str, title: str) -> str:
    prompt = f""" Gere uma lista de tópicos importantes do documento abaixo.

    Título: {title}

    Texto:
    {text[:10000]}

    Lista de tópicos em português no formato de bullet points:
    """

    max_retries = 5
    for attempt in range(max_retries):
        try:
            return llm.invoke(prompt).content
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "413" in error_str or "rate_limit" in error_str.lower() or "rate limit" in error_str.lower():
                wait = 2 ** attempt
                logger.warning(f"rate limit atingido em '{title}', tentativa {attempt+1}/{max_retries}, aguardando {wait}s...")
                time.sleep(wait)
            else:
                raise
    raise Exception(f"Falha após {max_retries} tentativas por rate limit: '{title}'")


class DocumentSkill(AgentSkill):
    name = "document"
    description = "Resumo e tópicos de documentos específicos"

    @property
    def tools(self):
        return [summarize_document, extract_document_bullets]

    @property
    def prompt_instructions(self):
        return """### Skill: document — Resumo e Tópicos de Documentos

**summarize_document(source_query)**: Gera um resumo completo de um documento pelo nome/título. APENAS quando o usuário pedir EXPLICITAMENTE "resumo" ou "sumário".

**extract_document_bullets(source_query)**: Extrai tópicos em bullet points de um documento pelo nome/título. APENAS quando o usuário pedir EXPLICITAMENTE "tópicos" ou "bullet points".

**Regras:**
- Use estas ferramentas SOMENTE quando o usuário pedir EXPLICITAMENTE "resumo", "sumário", "tópicos" ou "bullet points" de um documento específico pelo nome
- Para perguntas sobre o CONTEÚDO de documentos, use search_knowledge_base (skill rag)
- Prefixo da resposta: "Resumo do documento" ou "Tópicos extraídos do documento"
- Essas ferramentas já retornam a resposta final formatada, sem necessidade de chamar o LLM novamente"""


registry.register(DocumentSkill())
