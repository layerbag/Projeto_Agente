from sqlalchemy import text
from src.rag.vector_store import get_engine
import json

def criar_tabela():
    engine = get_engine()

    with engine.connect() as conn:
        conn.execute(text("""
         CREATE TABLE IF NOT EXISTS document_summaries (
            id SERIAL PRIMARY KEY,
            source TEXT UNIQUE NOT NULL,
            title TEXT NOT NULL,
            type TEXT NOT NULL,
            summary TEXT,
            topics jsonb default '[]',
            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ DEFAULT NOW()
        )
        """))

        conn.commit()

def salvar_sumario(source: str, title: str, doc_type: str, summary: str, topics: list | None = None):
    engine = get_engine()

    with engine.connect() as conn:
        conn.execute(text(r"""
            insert into document_summaries (source, title, type, summary, topics)
            values (:source, :title, :type, :summary, :topics\:\:jsonb)
            on conflict (source) do update set
                title = excluded.title,
                type = excluded.type,
                summary = excluded.summary,
                topics = excluded.topics,
                updated_at = now()
        """),{
            "source": source,
            "title": title,
            "type": doc_type,
            "summary": summary,
            "topics": json.dumps(topics or [], ensure_ascii=False)
        })

        conn.commit()

def buscar_sumario(source: str) -> dict | None:
    engine = get_engine()
    with engine.connect() as conn:
        row = conn.execute(text("""
            SELECT source, title, type, summary, chapters
            FROM document_summaries
            WHERE source = :source
        """), {"source": source}).fetchone()
        if row:
            return {
                "source": row[0], "title": row[1], "type": row[2],
                "summary": row[3], "chapters": row[4] or []
            }
    return None

def buscar_sumario_por_titulo(title: str) -> dict | None:
    engine = get_engine()
    with engine.connect() as conn:
        row = conn.execute(text("""
            SELECT source, title, type, summary
            FROM document_summaries
            WHERE title ILIKE :title
            LIMIT 1
        """), {"title": f"%{title}%"}).fetchone()
        if row:
            return {
                "source": row[0], "title": row[1], "type": row[2],
                "summary": row[3]
            }
    return None

def buscar_topicos_por_titulo(title: str) -> str | None:
    engine = get_engine()
    with engine.connect() as conn:
        row = conn.execute(text("""
            SELECT topics
            FROM document_summaries
            WHERE title ILIKE :title
            LIMIT 1
        """), {"title": f"%{title}%"}).fetchone()

        
        if row and row[0]:
            topics = f"### {title}\n" + "\n".join(f"- {topic}" for topic in row[0])
            return topics
    return None

def listar_sumarios() -> list[dict]:
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT source, title, type
            FROM document_summaries
            ORDER BY title
        """)).fetchall()
        return [{"source": r[0], "title": r[1], "type": r[2]} for r in rows]
