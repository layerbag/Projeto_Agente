from uuid import uuid4
import os
import tempfile
import sqlite3
import json
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from api.schemas import ChatRequest, ChatResponse, DeleteDocumentRequest
from src.rag.rag_chain import responder
from src.rag.ingestao import dividir_em_chunks, extrair_informacoes, extrair_transcricao
from src.utils.project_utils import load_pdf, yt2doc, normalizar_caminho, delete_indexed_document
from src.rag.vector_store import carregar_vector_store

app = FastAPI(title="Projeto Agente API")

# Libera chamadas da interface web e de outros clientes para a API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

sessions: dict[str, list[dict[str, str]]] = {}

# Serve os arquivos estáticos da interface em /static.
app.mount("/static", StaticFiles(directory="web"), name="static")

@app.get("/ui")
def ui():
    # Entrega a página principal da interface web.
    return FileResponse("web/index.html")

@app.get("/")
def health_check():
    # Endpoint simples para verificar se a API está no ar.
    return {"status": "ok", "message": "API do agente está rodando"}

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    # Usa uma sessão existente ou cria uma nova para manter o histórico da conversa.
    session_id = request.session_id or str(uuid4())

    if session_id not in sessions:
        sessions[session_id] = []

    sessions[session_id].append({
        "role": "user",
        "content": request.message,
    })

    try:
        # Envia a mensagem para a cadeia RAG/agente e associa a resposta à sessão.
        resposta = responder(request.message, session_id)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao executar o agente: {e}"
        )

    sessions[session_id].append({
        "role": "assistant",
        "content": resposta
    })

    return ChatResponse(
        session_id=session_id,
        response=resposta,
    )


def _normalize_role(role: str | None) -> str | None:
    # Converte nomes de papéis vindos de formatos diferentes para user/assistant.
    if not role:
        return None

    normalized = str(role).lower()
    if "human" in normalized or "user" in normalized:
        return "user"
    if "ai" in normalized or "assistant" in normalized:
        return "assistant"
    return None


def _decode_session_value(value):
    # Decodifica valores salvos no SQLite, aceitando JSON em bytes ou msgpack.
    if isinstance(value, (bytes, bytearray)):
        try:
            text = value.decode("utf-8")
            return json.loads(text)
        except Exception:
            pass

        try:
            import msgpack
            return msgpack.unpackb(value, raw=False)
        except Exception:
            return value

    return value


def _extract_messages(obj):
    # Percorre estruturas aninhadas do checkpoint procurando mensagens e seus papéis.
    try:
        import msgpack
        has_msgpack = True
    except Exception:
        msgpack = None
        has_msgpack = False

    def walk(item):
        if has_msgpack and isinstance(item, getattr(msgpack, "ExtType", type(None))):
            try:
                unpacked = msgpack.unpackb(item.data, raw=False)
                return walk(unpacked)
            except Exception:
                return []

        if isinstance(item, dict):
            role = item.get("type") or item.get("cls") or item.get("role") or item.get("name")
            content = item.get("content") or item.get("text") or item.get("message")

            messages = []
            if isinstance(content, str) and role:
                messages.append((role, content))

            for value in item.values():
                messages.extend(walk(value))
            return messages

        if isinstance(item, (list, tuple)):
            results = []
            for element in item:
                results.extend(walk(element))
            return results

        return []

    return walk(obj)


def _build_history_from_rows(rows):
    # Monta o histórico de chat a partir das linhas gravadas pelo checkpoint.
    history = []
    for channel, value in rows:
        decoded = _decode_session_value(value)
        for role_name, content in _extract_messages(decoded):
            role = _normalize_role(role_name) or _normalize_role(channel)
            if role and content:
                history.append({"role": role, "content": content})
    return history


def _format_preview(text: str | None) -> str | None:
    # Gera uma prévia curta da última mensagem para listar sessões.
    if not text:
        return None

    snippet = " ".join(text.strip().split())
    max_length = 80
    if len(snippet) <= max_length:
        return snippet
    return snippet[:max_length].rstrip() + "..."


@app.get("/sessions/{session_id}")
def get_session(session_id: str):
    rows = []
    try:
        # Primeiro tenta recuperar o histórico persistido pelo checkpoint do agente.
        conn = sqlite3.connect("checkpoints.sqlite")
        cur = conn.cursor()
        cur.execute(
            """
            SELECT channel, value
            FROM writes
            WHERE thread_id = ?
            ORDER BY rowid ASC
            """,
            (session_id,)
        )
        rows = cur.fetchall()
    except Exception:
        rows = []
    finally:
        try:
            conn.close()
        except Exception:
            pass

    history = _build_history_from_rows(rows)
    if history:
        return {"session_id": session_id, "history": history}

    # Se não houver checkpoint, usa o histórico temporário em memória.
    if session_id in sessions:
        return {"session_id": session_id, "history": sessions[session_id]}

    raise HTTPException(status_code=404, detail="Sessão não encontrada")


@app.delete("/sessions/{session_id}")
def delete_session(session_id: str):
    deleted = False

    try:
        # Remove a sessão tanto das escritas intermediárias quanto dos checkpoints.
        conn = sqlite3.connect("checkpoints.sqlite")
        cur = conn.cursor()
        cur.execute("DELETE FROM writes WHERE thread_id = ?", (session_id,))
        deleted_writes = cur.rowcount
        cur.execute("DELETE FROM checkpoints WHERE thread_id = ?", (session_id,))
        deleted_checkpoints = cur.rowcount
        conn.commit()
        conn.close()
        if deleted_writes > 0 or deleted_checkpoints > 0:
            deleted = True
    except Exception:
        try:
            conn.close()
        except Exception:
            pass

    # Também limpa o histórico em memória, quando existir.
    if session_id in sessions:
        del sessions[session_id]
        deleted = True

    if not deleted:
        raise HTTPException(status_code=404, detail="Sessão não encontrada")

    return {"status": "deleted", "session_id": session_id}


@app.post("/indexDoc")
async def index_docs(
    type: str = Form(...),
    url: str | None = Form(None),
    file: UploadFile | None = File(None),
):
    vector_store = carregar_vector_store()

    if type == "pdf":
        # Indexa um PDF enviado por upload, evitando duplicar fontes já indexadas.
        if file is None:
            raise HTTPException(status_code=400, detail="Arquivo PDF não enviado.")

        source = normalizar_caminho(file.filename or "uploaded.pdf")
        resultados = vector_store.get(where={"source": source})

        if len(resultados["ids"]) > 0:
            return {"message": f"{source} já indexado"}

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(await file.read())
            temp_path = tmp.name

        try:
            doc = load_pdf(temp_path,source.split('/')[-1])
            for page in doc:
                page.metadata["source"] = source
        finally:
            try:
                os.unlink(temp_path)
            except OSError:
                pass

        # Divide o documento em chunks e salva no vector store para uso no RAG.
        chunks = dividir_em_chunks(pdf_docs=doc)
        if chunks:
            vector_store.add_documents(chunks)
            return {"message": "Documento Indexado"}
        return {"message": "Documento Vazio"}

    if type == "YouTube":
        # Indexa a transcrição de um vídeo do YouTube, também evitando duplicidade.
        if not url:
            raise HTTPException(status_code=400, detail="URL do YouTube não informada.")
        url = normalizar_caminho(url)
        resultados = vector_store.get(where={"source": url})

        if len(resultados["ids"]) > 0:
            return {"message": f"{url} já indexado"}

        try:
            # Extrai metadados, transcrição, transforma em documento e persiste os chunks.
            videoId, title = extrair_informacoes(url)
            texto = extrair_transcricao(videoId)
            metadata = {"video_id": videoId, "title": title, "source": url, "type": "YouTube"}
            doc = yt2doc(texto, metadata)
            chunks = dividir_em_chunks(yt_docs=doc)
            if chunks:
                vector_store.add_documents(chunks)
                return {"message": "Documento Indexado"}
            return {"message": "Documento Vazio"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Erro ao indexar YouTube: {e}")

    raise HTTPException(status_code=400, detail="Tipo de indexação inválido.")


@app.get("/documents")
def list_documents():
    # Agrupa os chunks salvos no vector store por documento de origem.
    vector_store = carregar_vector_store()
    data = vector_store.get(include=["metadatas"])

    documents: dict[str, dict] = {}

    for metadata in data.get("metadatas", []) or []:
        metadata = metadata or {}
        source = str(metadata.get("source") or metadata.get("url") or metadata.get("title") or "")
        if not source:
            continue

        title = str(metadata.get("title") or source)
        doc_type = str(metadata.get("type") or "unknown")

        if source not in documents:
            documents[source] = {
                "source": source,
                "title": title,
                "type": doc_type,
                "count": 0,
            }

        documents[source]["count"] += 1

    return {"documents": list(documents.values())}


@app.delete("/documents")
def delete_document(request: DeleteDocumentRequest):
    result = delete_indexed_document(request.source)

    if not result["deleted"]:
        raise HTTPException(status_code=404, detail=result["message"])

    return result


@app.get("/sessions_db")
def sessions_db():
    # Lista as sessões persistidas no banco de checkpoints.
    conn = sqlite3.connect("checkpoints.sqlite")
    cur = conn.cursor()
    try:
        cur.execute(
            """
            SELECT thread_id, MAX(checkpoint_ns) as last_ns, COUNT(*) as count
            FROM checkpoints
            GROUP BY thread_id
            ORDER BY last_ns DESC
            """
        )
        rows = cur.fetchall()
    finally:
        conn.close()

    sessions_list = []
    for thread_id, last_ns, count in rows:
        preview = None
        try:
            # Usa a última mensagem do usuário como prévia da sessão.
            session_response = get_session(thread_id)
            user_messages = [item for item in session_response["history"] if item["role"] == "user"]
            if user_messages:
                preview = _format_preview(user_messages[-1]["content"])
        except Exception:
            preview = None

        sessions_list.append({
            "session_id": thread_id,
            "last_ns": last_ns,
            "count": count,
            "preview": preview,
        })

    return {"sessions": sessions_list}

@app.get("/logs")
def logs(lines: int = 200):
    log_path = "rag.log"

    if not os.path.exists(log_path):
        raise HTTPException(status_code=404, detail="Arquivo de log não encontrado")
    
    try:
        with open(log_path, "r", encoding="utf-8") as file:
            log_lines = file.readlines()

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao ler logs: {e}")
    
    return {
        "file": log_path,
        "lines": lines,
        "content": "".join(log_lines[-lines:]),
    }

