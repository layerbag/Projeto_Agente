from langchain.tools import tool
from sqlalchemy import text
from src.rag.rag_chain import busca_hibrida_pgvector, formatar_docs
from src.rag.vector_store import get_engine, carregar_vector_store
from src.mlops.metrics import measure_time
from src.mlops.observability import logger
from src.agent.skills.registry import AgentSkill, registry

from src.utils.project_utils import load_pdf, yt2doc, normalizar_caminho
from src.rag.ingestao import extrair_informacoes, extrair_transcricao, dividir_em_chunks
from src.rag.document_summarizer import summarize_document as bg_summarize_document
from src.rag.document_summary_store import salvar_sumario
import os


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


@tool
@measure_time("index_document")
def index_document(title: str) -> dict:
    """Indexa um novo documento (PDF local ou URL do YouTube) na base de conhecimento.
    Recebe a URL ou caminho do arquivo.
    """
    if not title or title.strip() == "":
        return {
            "result": "Por favor, peça ao usuário a URL do YouTube ou o caminho local do arquivo PDF que deseja indexar.",
            "final": False
        }
    
    title = normalizar_caminho(title.strip())
    vector_store = carregar_vector_store()
    
    try:
        resultados = vector_store.similarity_search("", filter={"source" : title})
        if len(resultados) > 0:
            return {"result": f"O documento '{title}' já está indexado.", "final": True}
            
        if "youtube.com" in title or "youtu.be" in title:
            videoId, title = extrair_informacoes(title)
            texto = extrair_transcricao(videoId)
            metadata = {"video_id": videoId, "title": title, "source": title, "type": "YouTube"}
            doc = yt2doc(texto, metadata)
            chunks = dividir_em_chunks(doc)
            
            if chunks:
                vector_store.add_documents(chunks)
                result_sum = bg_summarize_document(texto, str(title))
                salvar_sumario(title, str(title), "YouTube", result_sum["overall_summary"], result_sum["topics"])
                return {"result": f"Vídeo '{title}' indexado com sucesso.", "final": True}
            else:
                return {"result": f"Falha ao indexar vídeo: Transcrição vazia.", "final": True}
        else:
            # Assumimos PDF
            if not os.path.exists(title):
                return {"result": f"Arquivo não encontrado: {title}", "final": True}
            
            filename = os.path.basename(title)
            doc = load_pdf(title, filename)
            for page in doc:
                page.metadata["source"] = title
            
            chunks = dividir_em_chunks(doc)
            if chunks:
                vector_store.add_documents(chunks)
                full_text = "\n\n".join(page.page_content for page in doc)
                title = chunks[0].metadata["title"]
                result_sum = bg_summarize_document(full_text, title)
                salvar_sumario(title, title, "pdf", result_sum["overall_summary"], result_sum["topics"])
                return {"result": f"Documento '{filename}' indexado com sucesso.", "final": True}
            else:
                return {"result": f"Falha ao indexar documento: Arquivo vazio ou erro no chunking.", "final": True}
                
    except Exception as e:
        logger.error(f"Erro em index_document: {str(e)}")
        return {"result": f"Erro ao indexar '{title}': {str(e)}", "final": False}

@tool
def get_title(palavra_chave: str):
    """Busca o título exato de um documento na base de dados a partir de uma palavra-chave ou termo parcial.
    Sempre chame esta ferramenta primeiro ao receber um pedido para deletar ou excluir qualquer documento."""

    if not palavra_chave or palavra_chave.strip() == "":
        return {
            "result": "Por favor, peça ao usuário a URL do YouTube ou o caminho local do arquivo PDF que deseja indexar.",
            "final": False
        }

    resultados = busca_hibrida_pgvector(palavra_chave, limit=1)

    if not resultados:
        return {
            "result": f"Documento não encontrado",
            "final": True
        }
    
    titulo = resultados[0].metadata["title"]
    return {
        "result": (
            f"Título encontrado: '{titulo}'.\n"
            f"⚠️ AÇÃO OBRIGATÓRIA: Você DEVE agora perguntar ao usuário: "
            f"'Você deseja realmente deletar o documento \'{titulo}\'? (sim/não)'. "
            f"NÃO chame delete_document antes de receber confirmação explícita do usuário."
        ),
        "requires_confirmation": True,
        "document_title": titulo,
        "final": False
    }
    
@tool
def delete_document(title: str):
    """Deleta um documento da base de dados pelo seu título EXATO obtido de get_title.
    NUNCA chame esta ferramenta diretamente sem antes chamar get_title E pedir confirmação explícita ao usuário."""

    if not title or title.strip() == "":
        return {
            "result": "Por favor, peça ao usuário a URL do YouTube ou o caminho local do arquivo PDF que deseja indexar.",
            "final": False
        }
    
    engine = get_engine()

    try:
        
        with engine.begin() as conn:
            result = conn.execute(
                text("""
                    DELETE FROM langchain_pg_embedding
                    WHERE cmetadata->>'title' ILIKE :title

                """),
                {"title" : title}
            )

            deleted_count = result.rowcount

            if deleted_count > 0:
                return {
                "result": "Documento deletados com sucesso",
                "final": True
                }
            else:
                return {
                    "result": "Documento não pôde ser deletado",
                    "final": True
                }

    except Exception as e:
        logger.error(e)
        return {
            "result": f"Erro ao deletar documento: {str(e)}",
            "final": True
        }

class RAGSkill(AgentSkill):
    name = "rag"
    description = "Busca, indexa e deleta documentos (PDFs, YouTube)"

    @property
    def tools(self):
        return [search_knowledge_base, list_documents, index_document, delete_document, get_title]

    @property
    def prompt_instructions(self):
        return """### Skill: rag — Base de Conhecimento

**search_knowledge_base(query)**: Busca na base local (PDFs, YouTube). Use para QUALQUER pergunta sobre documentos indexados. Retorna trechos dos documentos mais relevantes.

**list_documents()**: Lista os títulos de todos os documentos disponíveis na base.

**index_document(title)**: Indexa um novo documento fornecido pelo usuário, seja URL do YouTube ou caminho para arquivo no PC.
SE o usuário pedir para indexar um documento MAS NÃO INFORMAR qual documento (URL ou caminho), NÃO chame a ferramenta, PERGUNTE antes qual o documento.
SE o usuário fornecer o caminho ou URL, chame essa ferramenta para indexar.
SE o usuário digitar mais de um documento, chame essa ferramenta UMA VEZ PARA CADA documento.

**get_title(palavra_chave)**: Busca na base de dados o título exato de um arquivo com base em palavras-chave.
- SEMPRE chame esta ferramenta primeiro ao receber um pedido para deletar/excluir qualquer documento, para descobrir o título exato do documento.

**delete_document(title)**: Deleta um documento que já está indexado na base, usando seu título exato.
- ⛔ PROIBIDO chamar esta ferramenta sem confirmação explícita do usuário na mesma conversa.
- O fluxo obrigatório para deleção é:
  1. Chamar `get_title(palavra_chave)` com a palavra-chave fornecida.
  2. **PARAR** e apresentar ao usuário a pergunta de confirmação retornada por `get_title` (campo `result`). Aguardar a resposta do usuário.
  3. SOMENTE se o usuário responder confirmando (ex: "sim", "pode deletar", "confirmo"), chamar `delete_document(title)` com o título exato do campo `document_title` retornado por `get_title`.
  4. Se o usuário recusar ou se o documento não for o correto, perguntar se ele deseja fornecer outro termo.

**Regras:**
- Para QUALQUER pergunta sobre o conteúdo dos documentos, use **search_knowledge_base** primeiro
- Se search_knowledge_base NÃO responder a pergunta de forma direta e completa com os trechos retornados, chame web_search automaticamente para complementar
- SOMENTE chame a delete_document(title) se o usuário confirmar explicitamente que quer excluir o documento retornado pela get_title()
- Prefixo da resposta: "Com base nos documentos indexados"
- Cite o título/fonte dos documentos na resposta"""

registry.register(RAGSkill())
