# ============================================================
# agent_graph.py — Grafo do agente com LangGraph
# Fluxo: START → agent → tools_condition
#   → se tool_calls → tool_node → verifica_final
#     → "final_response" (se tool retornou {"final": True}) → END
#     → "agent" (se não) → processa resposta → END
#   → se sem tool_calls → END
# ============================================================

from langchain_core.messages import AIMessage, ToolMessage, SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END, MessagesState, START
from langgraph.prebuilt import ToolNode, tools_condition
from uuid import uuid4
from src.agent.tools import tools_list
from src.agent.prompts import SYSTEM_PROMPT
from src.rag.rag_chain import groq, gemini, memory
from src.mlops.observability import observe_query, logger
from src.mlops.metrics import measure_time
import aiosqlite
import json
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

# --- LLM com tools + fallback ---
groq_with_tools = groq.bind_tools(tools_list)
gemini_with_tools = gemini.bind_tools(tools_list)
llm_tools_fallback = groq_with_tools.with_fallbacks([gemini_with_tools])

# ============================================================
# STREAM — versão assíncrona com SSE (Server-Sent Events)
# ============================================================
async def stream_agent_events(message: str, session_id: str | None):
    config = {"configurable": {"thread_id": session_id or str(uuid4())}}
    sid = config["configurable"]["thread_id"]
    active_node = None  # rastreia qual nó está executando

    async with aiosqlite.connect("checkpoints.sqlite") as aconn:
        async_memory = AsyncSqliteSaver(conn=aconn)
        async_builder = StateGraph(MessagesState)
        async_builder.add_node("agent", agent_node)
        async_builder.add_node("tools", tool_node)
        async_builder.add_edge(START, "agent")
        async_builder.add_conditional_edges("agent", tools_condition)
        async_builder.add_node("final_response", final_response)
        async_builder.add_conditional_edges("tools", verifica_final, {
            "agent": "agent",
            "final_response": "final_response"
        })
        async_builder.add_edge("final_response", END)
        async_graph = async_builder.compile(checkpointer=async_memory)

        async for event in async_graph.astream_events(
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
            elif kind == "on_chain_end" and name in ("agent", "tools", "final_response"):
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
                elif name == "final_response":
                    output = event["data"].get("output", {})
                    msgs = output.get("messages", []) if isinstance(output, dict) else []
                    if msgs:
                        last = msgs[-1]
                        content = last.content if hasattr(last, "content") else str(last)
                        if content:
                            yield {"type": "done", "content": content, "session_id": sid}
                active_node = None
                continue

            # --- Eventos streaming do agent (pensamento + tokens) ---
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
                    yield {"type": "tool", "content": f"🔍 Usando: **{name}**..."}
                elif kind == "on_tool_end":
                    yield {"type": "tool_result", "content": f"✅ **{name}** concluída"}


# ============================================================
# NÓ AGENT — async (chama o LLM com as tools)
# ============================================================
@measure_time("agent_node")
async def agent_node(state: MessagesState) -> dict:
    # Extrai e observa a query do usuário
    user_msg = state["messages"][-1].content if state["messages"] else ""
    observe_query(user_msg)

    # Invoca o LLM com system prompt + últimas 10 mensagens
    messages = state["messages"][-10:]
    response = await llm_tools_fallback.ainvoke([
        SystemMessage(content=SYSTEM_PROMPT)
    ] + messages)

    # Log do que o agente fez (chamou tool ou respondeu texto)
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
# NÓ DE ROTEAMENTO — decide se vai para final_response ou volta p/ agent
# ============================================================
def verifica_final(state):
    """Se a última tool retornou JSON com {"final": true},
    vai direto para final_response sem chamar o LLM de novo."""
    messages = state["messages"]
    last_msg = messages[-1]

    if not isinstance(last_msg, ToolMessage):
        return "agent"

    content = last_msg.content
    try:
        if isinstance(content, str):
            content = json.loads(content)
        if isinstance(content, dict) and content.get("final") is True:
            return "final_response"
    except Exception:
        pass

    return "agent"


# ============================================================
# NÓ FINAL — extrai o result do JSON e devolve como AIMessage
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
# TOOL NODE — executa as tools chamadas pelo LLM
# ============================================================
tool_node = ToolNode(tools_list)


# ============================================================
# GRAFO SÍNCRONO (para API REST não-streaming)
# ============================================================
builder = StateGraph(MessagesState)
builder.add_node("agent", agent_node_sync)
builder.add_node("tools", tool_node)
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tools_condition)
builder.add_node("final_response", final_response)
builder.add_conditional_edges("tools", verifica_final, {
    "agent": "agent",
    "final_response": "final_response"
})
builder.add_edge("final_response", END)

agent_graph = builder.compile(checkpointer=memory)
