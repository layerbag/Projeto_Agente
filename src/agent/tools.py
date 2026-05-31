import time
from langchain.tools import tool
from sqlalchemy import text
from langchain_community.tools import DuckDuckGoSearchRun
from src.rag.rag_chain import busca_hibrida_pgvector, formatar_docs, llm
from src.rag.vector_store import get_engine
from src.mlops.metrics import measure_time
from src.mlops.observability import logger
from src.rag.document_summary_store import buscar_topicos_por_titulo, buscar_sumario_por_titulo

# tool 1: busca na base de conhecimento
@tool
@measure_time("search_knowledge_base")
def search_knowledge_base(query: str) -> dict:
    """Busca documentos relevantes na base de conhecimento usando busca híbrida (vetorial + texto). 
    Use para perguntas sobre PDFs, vídeos do YouTube, tutoriais, manuais."""

    docs = busca_hibrida_pgvector(query=query, limit=5)
    logger.info(f"search_knowledge_base encontrou {len(docs)} documentos para: {query[:80]}")
    return {
        "result": formatar_docs(docs),
        "final": False
    }

# tool 2: busca na web
@tool
@measure_time("web_search")
def web_search(query: str) -> dict:
    """Busca informações atualizadas na web. Use para notícias, preços atuais, 
    informações que podem ter mudado desde a última indexação dos documentos."""

    search = DuckDuckGoSearchRun()
    result = search.run(query)
    logger.info(f"web_search retornou {len(result)} caracteres para: {query[:80]}")
    return {
        "result": result,
        "final": False
    }

# tool 3: Listar documentos disponíveis
@tool
@measure_time("list_documents")
def list_documents() -> dict:
    """Lista todos os documentos disponíveis na base de conhecimento."""
    engine = get_engine()

    try:
        with engine.connect() as conn:
            results = conn.execute(text("""
                SELECT DISTINCT
                    cmetadata->>'title' AS title,
                    cmetadata->>'type' AS type
                FROM langchain_pg_embedding
                ORDER BY title
            """))
            rows = results.fetchall()
            logger.info(f"list_documents encontrou {len(rows)} documentos")
            documents = "".join(f"Documento: {row.title} Tipo: {row.type}\n" for row in rows)
            return {
                "results": documents,
                "final": False
            }
    except Exception as e:
        logger.error(f"list_documents falhou: {e}")
        return {
            "results": str(e),
            "final": False
        }

# tool 4: resumir documentos
@tool
@measure_time("get_document_summary")
def get_document_summary(source_query: str) -> dict:
    """Gera um resumo completo de um documento específico da base de conhecimento.
    Use quando o usuário pedir um resumo de um PDF, vídeo, ou documento pelo nome/título.
    O parâmetro source deve ser o título ou a fonte do documento."""

    resultados = busca_hibrida_pgvector(source_query, limit=1)

    if not resultados:
        return {
            "result": f"Documento com título/source '{source_query}' não encontrado",
            "final": True
        }

    source = resultados[0].metadata["title"]

    cached = buscar_sumario_por_titulo(source)
    print(cached)
    if cached and cached["summary"]:
        return {
            "result": cached["summary"],
            "final": True
        }
    
    else:
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
        doc_type = metadata.get("type", "desconhecido")

        logger.info(f"get_document_summary: {title} ({doc_type}), {len(full_text)} caracteres, {len(rows)} chunks")

        MAX_CHARS = 50000

        if len(full_text) <= MAX_CHARS:
            return {
                "result":_summarize(full_text, title),
                "final": True
            }
        
        chunks = [full_text[i:i+MAX_CHARS] for i in range(0, len(full_text), MAX_CHARS)]
        summaries = [
            _summarize(c, f"{title} (parte {i+1}/{len(chunks)})")
            for i, c in enumerate(chunks)
        ]

        final = _summarize("\n\n".join(summaries), title)

        return {
            "result":final,
            "final": True
        }

def _summarize(text: str, title: str) -> str:
    """Sumariza um texto usando o LLM com retry em caso de rate limit."""

    prompt = f""" Resuma o documento abaixo de forma clara e concisa.

    Título: {title}

    Texto:
    {text[:10000]}

    Resumo em português, 3-5 parágrafos:
    """

    max_retries = 5
    for attempt in range(max_retries):
        try:
            return llm.invoke(prompt).content #type: ignore
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
@measure_time("get_document_topics")
def get_document_topics(source_query: str) -> dict:
    """Gera uma bullet list de um documento específico da base de conhecimento.
    Use quando o usuário pedir tópicos de um PDF, vídeo, ou documento pelo nome/título.
    O parâmetro source deve ser o título ou a fonte do documento."""

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
    doc_type = metadata.get("type", "desconhecido")

    logger.info(f"get_document_topics: {title} ({doc_type}), {len(full_text)} caracteres, {len(rows)} chunks")

    MAX_CHARS = 50000

    if len(full_text) <= MAX_CHARS:
        return {
            "result":_topics(full_text, title),
            "final": True
        }
    
    
    chunks = [full_text[i:i+MAX_CHARS] for i in range(0, len(full_text), MAX_CHARS)]
    topics = [
        _topics(c, f"{title} (parte {i+1}/{len(chunks)})")
        for i, c in enumerate(chunks)
    ]

    final = "\n\n".join(topics)

    return {
        "result":final,
        "final":True
    }

def _topics(text: str, title: str) -> str:
    """gera tópicos de um texto usando o LLM com retry em caso de rate limit."""

    prompt = f""" Gere uma lista de tópicos importantes do documento abaixo.

    Título: {title}

    Texto:
    {text[:10000]}

    Lista de tópicos em português no formato de bullet points:
    """

    max_retries = 5
    for attempt in range(max_retries):
        try:
            return llm.invoke(prompt).content #type: ignore
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "413" in error_str or "rate_limit" in error_str.lower() or "rate limit" in error_str.lower():
                wait = 2 ** attempt
                logger.warning(f"rate limit atingido em '{title}', tentativa {attempt+1}/{max_retries}, aguardando {wait}s...")
                time.sleep(wait)
            else:
                raise
    raise Exception(f"Falha após {max_retries} tentativas por rate limit: '{title}'")


tools_list = [search_knowledge_base,web_search, list_documents, get_document_summary, get_document_topics]