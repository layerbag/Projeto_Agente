from typing import TypedDict, List, Annotated
from rank_bm25 import BM25Okapi # type: ignore
from langchain_classic.retrievers.document_compressors import LLMChainExtractor
from sentence_transformers import CrossEncoder
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, BaseMessage, AIMessage
from dotenv import load_dotenv
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
from src.rag.vector_store import carregar_vector_store
from src.mlops.metrics import measure_time
from src.mlops.observability import (
    observe_query,
    observe_docs,
    observe_response
)

load_dotenv()

_bm25 = None
_bm25_docs = None

conn = sqlite3.connect("checkpoints.sqlite", check_same_thread=False)
memory = SqliteSaver(conn)

groq = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
gemini = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

llm = groq.with_fallbacks([gemini])

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


class RAGState(TypedDict, total=False):
    query: str
    rewritten_query: str
    context: str
    response: str
    filter: dict
    messages: Annotated[list[BaseMessage], add_messages]


def inicializar_bm25():
    global _bm25
    global _bm25_docs

    if _bm25 is not None:
        return _bm25, _bm25_docs
    
    vector_store = carregar_vector_store()

    data = vector_store.get()

    textos = data["documents"]
    metadatas = data["metadatas"]

    docs = []

    for texto, metadata in zip(textos, metadatas):
        fake_doc = type("Doc",(),{})()
        fake_doc.page_content = texto
        fake_doc.metadata = metadata
        docs.append(fake_doc)
    
    tokenized_docs = [
        doc.page_content.lower().split() for doc in docs
    ]

    _bm25 = BM25Okapi(tokenized_docs)
    _bm25_docs = docs

    return _bm25, _bm25_docs

@measure_time("retrieve_docs")
def retrieve_docs(state: RAGState) -> dict:
    
    query = state.get("rewritten_query") or state["query"]
    filtro = state.get("filter")

    vector_store = carregar_vector_store()

    # =====================================================
    # 1. VECTOR SEARCH COM MMR
    # =====================================================

    if filtro:
        semantic_docs = vector_store.max_marginal_relevance_search(
            query=query,
            k=10,
            fetch_k=20,
            filter=filtro
        )
    else:
        semantic_docs = vector_store.max_marginal_relevance_search(
            query=query,
            k=10,
            fetch_k=20
        )

    
    # =====================================================
    # 2. BM25 SEARCH
    # =====================================================

    bm25, bm25_docs = inicializar_bm25()

    tokenized_query = query.lower().split()

    bm25_results = bm25.get_top_n(
        tokenized_query,
        bm25_docs,
        n=10
    )

    # =====================================================
    # 3. MERGE HÍBRIDO
    # =====================================================

    combined_docs = semantic_docs + bm25_results

    # remove duplicados
    unique_docs = list({
        doc.page_content: doc
        for doc in combined_docs
    }.values())

    

    # =====================================================
    # 4. RE-RANKING
    # =====================================================

    if not unique_docs:
        return {
            "context": "Nenhum documento relevante encontrado."
        }

    pairs = [
        (query, doc.page_content)
        for doc in unique_docs
    ]

    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(unique_docs, scores),
        key=lambda x: x[1],
        reverse=True
    )

    reranked_docs = [
        doc
        for doc, score in ranked[:5]
    ]

    
    # =====================================================
    # 5. CONTEXT COMPRESSION
    # =====================================================

    compressor_groq = LLMChainExtractor.from_llm(groq)
    compressor_gemini= LLMChainExtractor.from_llm(gemini)

    try:
        compressed_docs = compressor_groq.compress_documents(
            reranked_docs,
            query
        )
    except Exception:
        compressed_docs = compressor_gemini.compress_documents(
            reranked_docs,
            query
        )
    # =====================================================
    # 6. FORMATAR CONTEXTO
    # =====================================================

    if not compressed_docs:
        compressed_docs = reranked_docs

    
    context = formatar_docs(compressed_docs)

    observe_docs(query, reranked_docs)

    return {
        "context": context
    }

@measure_time("generate_answer")
def generate_answer(state: RAGState):
    prompt = ChatPromptTemplate.from_messages([
        ("system",
        """Você é um assistente que responde APENAS com base no contexto.
         
        Contexto:
        {context}
        """),
        MessagesPlaceholder("messages"),
        ("human","{query}")
    ])

    chain = prompt | llm | StrOutputParser()

    response = chain.invoke({
        "context": state["context"],
        "query": state["query"],
        "messages": state.get("messages",[])
    })

    observe_response(response)

    return {
        "response": response,
        "messages": state.get("messages", []) + [
            HumanMessage(content=state["query"]),
            AIMessage(content=response)
        ]
    }

def contextualize_question(state: RAGState):
    query = state["query"]
    messages = state.get("messages", [])

    history = "\n".join(
        f"{msg.type}: {msg.content}"
        for msg in messages[-8:]
    )

    prompt = f"""
    Data a conversa anterior e a pergunta atual, reescreva a pergunta atual para que ela seja completa e possa ser usada em uma busca vetorial.

    Histórico da conversa:
    {history}

    Pergunta atual: {query}

    retorne apenas a pergunta reescrita.
    """

    rewritten = llm.invoke(prompt)
    print(rewritten.content)
    return {
        "rewritten_query": rewritten.content
    }

graph = StateGraph(RAGState)

graph.add_node("contextualize", contextualize_question)
graph.add_node("retrieve", retrieve_docs)
graph.add_node("generate", generate_answer)

graph.set_entry_point("contextualize")
graph.add_edge("contextualize", "retrieve")
graph.add_edge("retrieve", "generate")
graph.add_edge("generate", END)

app = graph.compile(checkpointer=memory)
    
# Formatar os documentos recuperados para o prompt
def formatar_docs(docs): # type: ignore
    """junta os chunks em um único texto formatado para o prompt."""

    textos = []

    for doc in docs:
        tipo = doc.metadata.get("type", "desconhecido") # type: ignore
        titulo = doc.metadata.get("title", "Sem título") # type: ignore
        source = doc.metadata.get("source", "desconhecido") # type: ignore

        texto = f"""
            Título: {titulo}
            Fonte: {source}
            Tipo: {tipo}

            Conteúdo: 
            {doc.page_content}
            """
        textos.append(texto)
    
    return "\n\n---\n\n".join(textos)

# Recuperar os documentos relevantes para a query usando o vector store
def recuperar_docs(state: RAGState) -> dict:
    """
    Recebe o estado completo e retorna apenas o que será atualizado.
    """
    query = state["query"]
    docs = carregar_vector_store().similarity_search(query, k=4)

    return {"context": formatar_docs(docs)}

def responder (request:str, session:str) -> str:
    config = {"configurable": {"thread_id": session}}

    response = app.invoke(
        {"query": request},
            config = config
    )

    return response['response']

# main para teste
if __name__ == "__main__":
    perguntas = [
        "O que acontece no vídeo?",
        "Qual é a reação do apresentador?",
        "A quem ele se refere quando fala 'oque que você fez?'?"
        "Sobre qual jogo é o vídeo?"
    ]

    for pergunta in perguntas:
        print(f"\nPergunta: {pergunta}")
        resposta = app.invoke({"query": pergunta})
        print("\nresposta:\n", resposta)