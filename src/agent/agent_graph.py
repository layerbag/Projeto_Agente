# ============================================================
# agent_graph.py — Grafo do agente com LangGraph
# Fluxo: START → agent → tools_condition
#   → se tool_calls → tools → route_after_tools
#     → "final_response"        (final=True)        → END  [sem LLM]
#     → "confirmation_response" (requires_confirmation=True) → END  [sem LLM]
#     → "agent"                 (fluxo normal)       → loop
#   → se sem tool_calls → END
# ============================================================

from langchain_core.messages import AIMessage, ToolMessage, SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END, MessagesState, START
from langgraph.prebuilt import ToolNode, tools_condition
from uuid import uuid4
from src.rag.rag_chain import groq, gemini, memory
from src.mlops.observability import observe_query, logger
from src.mlops.metrics import measure_time
import aiosqlite
from src.agent.skills import registry
import json
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

tools_list = registry.all_tools()
SYSTEM_PROMPT = f"""Você é um assistente inteligente com acesso a múltiplas ferramentas.

## Ferramentas disponíveis:
{registry.all_prompt_instructions()}

## Regras:
- SEMPRE escolha a ferramenta adequada para cada pergunta
- Se uma ferramenta retornar resultado COMPLETO que responda diretamente à pergunta, não chame outra
- Se o resultado for parcial, insuficiente ou não responder diretamente, chame web_search como complemento
- Responda em português
- Cite as fontes das informações"""

# --- LLM com tools + fallback ---
groq_with_tools = groq.bind_tools(tools_list)
gemini_with_tools = gemini.bind_tools(tools_list)
llm_tools_fallback = groq_with_tools.with_fallbacks([gemini_with_tools])

# ============================================================
# TOOL NODE — executa as tools chamadas pelo LLM
# ============================================================
tool_node = ToolNode(tools_list)


# ============================================================
# BLOQUEIO DE DELEÇÃO — impede delete_document se usuário recusou
# ============================================================
PALAVRAS_NEGACAO = {"não", "nao", "n", "nope", "cancelar", "cancela", "negativo", "nem", "nah"}


def _extract_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(b.get("text", "") for b in content if isinstance(b, dict))
    return str(content)


def _user_refused_deletion(messages: list) -> bool:
    """Verifica se a última resposta do usuário recusou uma confirmação de deleção."""
    for i in range(len(messages) - 2, -1, -1):
        msg = messages[i]
        try:
            text = _extract_text(msg.content).lower()
            if isinstance(msg, AIMessage) and "tem certeza que quer deletar" in text:
                ultima = messages[-1]
                if isinstance(ultima, HumanMessage):
                    texto = _extract_text(ultima.content).strip().lower()
                    return any(p in texto for p in PALAVRAS_NEGACAO) or texto.startswith("n")
        except Exception:
            continue
    return False


# ============================================================
# NÓ AGENT — async (chama o LLM com as tools)
# ============================================================
@measure_time("agent_node")
async def agent_node(state: MessagesState) -> dict:
    user_msg = state["messages"][-1].content if state["messages"] else ""
    observe_query(user_msg)

    # BLOQUEIO PREVENTIVO: se usuário recusou deleção, nem chama o LLM
    if _user_refused_deletion(state["messages"]):
        logger.info("Usuário recusou deleção — resposta direta sem LLM")
        return {"messages": [
            AIMessage(content="Deleção cancelada. O documento não foi removido.")
        ]}

    messages = state["messages"][-10:]
    response = await llm_tools_fallback.ainvoke([
        SystemMessage(content=SYSTEM_PROMPT)
    ] + messages)

    # SAFETY NET: intercepta delete_document mesmo se o LLM for chamado
    if (
        hasattr(response, "tool_calls") and response.tool_calls
        and any(tc["name"] == "delete_document" for tc in response.tool_calls)
        and _user_refused_deletion(state["messages"])
    ):
        logger.info("Usuário recusou deleção — bloqueando delete_document (safety net)")
        response = AIMessage(content="Deleção cancelada. O documento não foi removido.")

    if hasattr(response, "tool_calls") and response.tool_calls:
        tool_names = [tc["name"] for tc in response.tool_calls]
        logger.info(f"Agent chamou tools: {tool_names}")
    else:
        content = response.content
        if isinstance(content, list):
            content = "".join(b.get("text", "") for b in content if isinstance(b, dict))
        if content:
            logger.info(f"Agent respondeu: {content[:200]}...")
        else:
            logger.warning("Agent respondeu com conteúdo vazio!")

    return {"messages": [response]}


# --- Versão síncrona (para uso com .invoke) ---
def agent_node_sync(state: MessagesState) -> dict:
    import asyncio
    return asyncio.run(agent_node(state))


# ============================================================
# NÓ DE ROTEAMENTO — 3 rotas após execução de tools
# ============================================================
def route_after_tools(state) -> str:
    """Roteia após a execução de uma tool:
    - "final_response"        se final=True        (sem LLM)
    - "confirmation_response" se requires_confirmation=True (sem LLM)
    - "agent"                 caso contrário        (continua o loop)
    """
    messages = state["messages"]
    last_msg = messages[-1]

    if not isinstance(last_msg, ToolMessage):
        return "agent"

    try:
        content = last_msg.content
        
        if isinstance(content, str):
            content = json.loads(content)
        
        if isinstance(content, dict):
            if content.get("final") is True:
                return "final_response"
            if content.get("requires_confirmation") is True:
                return "confirmation_response"
    except Exception:
        pass

    return "agent"


# ============================================================
# NÓ FINAL — extrai o result do JSON e devolve como AIMessage
# (usado quando final=True — bypass total do LLM)
# ============================================================
def final_response(state):
    """Pega o resultado da tool que retornou {"result": "...", "final": True}
    e devolve direto como resposta, sem passar pelo LLM."""
    last_tool = state["messages"][-1]
    content = last_tool.content
    if isinstance(content, str):
        content = json.loads(content)

    return {
        "messages": [AIMessage(content=content["result"])]
    }


# ============================================================
# NÓ DE CONFIRMAÇÃO — apresenta a pergunta ao usuário
# sem chamar o LLM (usado quando requires_confirmation=True)
# ============================================================
def confirmation_response(state):
    """Extrai a mensagem de confirmação preparada pela tool e devolve
    diretamente ao usuário, sem passar pelo LLM."""
    last_tool = state["messages"][-1]
    content = last_tool.content
    if isinstance(content, str):
        content = json.loads(content)

    return {
        "messages": [AIMessage(content=f"tem certeza que quer deletar o documento: \"{content["document_title"]}\"?")]
    }


# ============================================================
# GRAFO ASYNC — pré-compilado uma única vez no startup
# (inicializado pelo lifespan do FastAPI via init_async_graph)
# ============================================================
_async_conn: aiosqlite.Connection | None = None
_async_graph = None


async def init_async_graph():
    """Inicializa a conexão SQLite persistente e compila o grafo async.
    Deve ser chamado uma única vez no startup da aplicação (lifespan do FastAPI).
    """
    global _async_conn, _async_graph

    _async_conn = await aiosqlite.connect("checkpoints.sqlite")
    async_memory = AsyncSqliteSaver(conn=_async_conn)

    async_builder = StateGraph(MessagesState)
    async_builder.add_node("agent", agent_node)
    async_builder.add_node("tools", tool_node)
    async_builder.add_node("final_response", final_response)
    async_builder.add_node("confirmation_response", confirmation_response)
    async_builder.add_edge(START, "agent")
    async_builder.add_conditional_edges("agent", tools_condition)
    async_builder.add_conditional_edges("tools", route_after_tools, {
        "agent": "agent",
        "final_response": "final_response",
        "confirmation_response": "confirmation_response",
    })
    async_builder.add_edge("final_response", END)
    async_builder.add_edge("confirmation_response", END)

    _async_graph = async_builder.compile(checkpointer=async_memory)
    logger.info("Grafo async compilado e pronto.")


async def close_async_graph():
    """Fecha a conexão SQLite do grafo async.
    Deve ser chamado no shutdown da aplicação (lifespan do FastAPI).
    """
    global _async_conn
    if _async_conn is not None:
        await _async_conn.close()
        _async_conn = None
        logger.info("Conexão SQLite do grafo async encerrada.")


# ============================================================
# STREAM — versão assíncrona com SSE (Server-Sent Events)
# ============================================================
async def stream_agent_events(message: str, session_id: str | None):
    if _async_graph is None:
        raise RuntimeError(
            "Grafo async não inicializado. "
            "Certifique-se de que init_async_graph() foi chamado no lifespan."
        )

    config = {"configurable": {"thread_id": session_id or str(uuid4())}}
    sid = config["configurable"]["thread_id"]
    active_node = None

    async for event in _async_graph.astream_events(
        {"messages": [HumanMessage(content=message)]},
        config=config,
        version="v2",
    ):
        kind = event["event"]
        name = event.get("name", "")
        # --- Início de nó: marca qual nó está ativo ---
        if kind == "on_chain_start" and name in ("agent", "tools"):
            active_node = name

        # --- Fim de nó: emite o resultado final "done" ---
        elif kind == "on_chain_end" and name in ("agent", "tools", "final_response", "confirmation_response"):
            if name == "agent":
                output = event["data"].get("output", {})
                msgs = output.get("messages", []) if isinstance(output, dict) else []
                if msgs:
                    last = msgs[-1]
                    content = last.content if hasattr(last, "content") else str(last)
                    if isinstance(content, list):
                        content = "".join(b.get("text", "") for b in content if isinstance(b, dict))
                    if content and not (hasattr(last, "tool_calls") and last.tool_calls):
                        yield {"type": "done", "content": content, "session_id": sid}

            elif name in ("final_response", "confirmation_response"):
                output = event["data"].get("output", {})
                msgs = output.get("messages", []) if isinstance(output, dict) else []
                if msgs:
                    last = msgs[-1]
                    content = last.content if hasattr(last, "content") else str(last)
                    if content:
                        yield {"type": "done", "content": content, "session_id": sid}
            active_node = None
            continue

        # --- Eventos streaming do agent (tokens) ---
        if active_node == "agent":
            if kind == "on_chat_model_start":
                yield {"type": "thinking", "content": "Analisando..."}

            elif kind == "on_chat_model_stream":
                chunk = event["data"].get("chunk", "")
                if hasattr(chunk, "content") and chunk.content:
                    has_tool_calls = hasattr(chunk, "tool_call_chunks") and chunk.tool_call_chunks
                    if not has_tool_calls:
                        yield {"type": "token", "content": chunk.content}

        # --- Eventos da tool (início/fim) ---
        elif active_node == "tools":
            if kind == "on_tool_start":
                yield {"type": "tool", "content": f"Usando: **{name}**..."}

            elif kind == "on_tool_end":
                yield {"type": "tool_result", "content": f"**{name}** concluída"}
