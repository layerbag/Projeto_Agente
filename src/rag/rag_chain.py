from pydantic._internal._validators import max_length_validator
from langchain_core.output_parsers import json
import json
from typing import TypedDict, List, Annotated
from rank_bm25 import BM25Okapi # type: ignore
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
from langchain_core.documents import Document
from src.rag.vector_store import get_engine, COLLECTION_NAME, get_embeddings
from sqlalchemy import text
from llmlingua import PromptCompressor
from transformers import AutoTokenizer
from src.mlops.metrics import measure_time
from src.mlops.observability import (
    observe_query,
    observe_docs,
    observe_response,
    logger
)

load_dotenv()

_llm_lingua_tokenizer = None

conn = sqlite3.connect("checkpoints.sqlite", check_same_thread=False)
memory = SqliteSaver(conn)

groq = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
gemini = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

llm = groq.with_fallbacks([gemini])
llm_lingua = PromptCompressor(
    model_name="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank", 
    use_llmlingua2=True
)


reranker = CrossEncoder(
    "BAAI/bge-reranker-v2-m3"
)


class RAGState(TypedDict):
    query: str
    rewritten_query: str
    context: str
    response: str
    filter: dict
    messages: Annotated[list[BaseMessage], add_messages]


def _get_llm_lingua_tokenizer():
    global _llm_lingua_tokenizer

    if _llm_lingua_tokenizer is None:
        _llm_lingua_tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank"
        )
    
    return _llm_lingua_tokenizer

def busca_hibrida_pgvector(query: str, limit: int = 5, rrf_k: int = 60) -> list[Document]:
    """
    Executa busca híbrida nativa no PostgreSQL (pgvector + Full Text Search)
    utilizando Reciprocal Rank Fusion (RRF) em uma única consulta SQL.
    """

    # Carregar e gerar os embeddings da query
    embeddings_model = get_embeddings()
    query_embedding = embeddings_model.embed_query(query)

    #Formatar para o pgvector
    query_embedding_str = f"[{','.join(map(str, query_embedding))}]"

    # Consulta SQL RRF
    sql_query = text(r"""
        WITH vector_search AS (
            SELECT
                e.id,
                ROW_NUMBER() OVER (ORDER BY e.embedding <=> CAST(:query_embedding AS vector)) as rank
            FROM langchain_pg_embedding e
            JOIN langchain_pg_collection c ON e.collection_id = c.uuid
            WHERE c.name = :collection
            LIMIT 20
        ),
        fts_search AS (
            SELECT
                e.id,
                ROW_NUMBER() OVER (
                    ORDER BY ts_rank_cd(to_tsvector('portuguese', e.document), plainto_tsquery('portuguese', :query)) DESC
                ) as rank
            FROM langchain_pg_embedding e
            JOIN langchain_pg_collection c ON e.collection_id = c.uuid
            WHERE c.name = :collection AND to_tsvector('portuguese', e.document) @@ plainto_tsquery('portuguese', :query)
            LIMIT 20
        )
        SELECT
            e.document,
            e.cmetadata,
            COALESCE(1.0 / (:rrf_k + v.rank), 0.0) + COALESCE(1.0 / (:rrf_k + f.rank), 0.0) as rrf_score
            FROM vector_search v
            FULL OUTER JOIN fts_search f ON v.id = f.id
            JOIN langchain_pg_embedding e ON e.id = COALESCE(v.id, f.id)
            ORDER BY rrf_score DESC
            LIMIT :limit;
    """)

    results = []

    engine = get_engine()

    try:
        with engine.connect() as conn:
            
            result = conn.execute(
                sql_query,
                {
                    "query_embedding" : query_embedding_str,    # vetor da busca semântica
                    "collection" : COLLECTION_NAME,    # Nome da coleção
                    "query" : query,              # Query para FTS
                    "rrf_k" : rrf_k,              # Constante RRF k para vetor
                    "limit" : limit,              # Limite final de documentos
                }
            )

            for row in result:
                document_text = row[0]
                metadata = row[1]
                rrf_score = row[2]

                # Se o metadado veio como string do banco, converte para dicionário Python
                if isinstance(metadata, str):
                    try:
                        metadata = json.loads(metadata)
                    except Exception:
                        metadata = {}

                metadata["rrf_score"] = rrf_score

                tokenizer = _get_llm_lingua_tokenizer()

                tokens = tokenizer.encode(document_text, truncation=True, max_length=480)

                truncated_text = tokenizer.decode(tokens, skip_specials_tokens=True)

                compressed_text = llm_lingua.compress_prompt([truncated_text], question=query)

                logger.info(f"\nTokens originais: {compressed_text["origin_tokens"]}\ntokens compressados:{compressed_text["compressed_tokens"]}")

                results.append(Document(page_content=compressed_text["compressed_prompt"], metadata=metadata)) #type: ignore

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[ERRO BUSCA HÍBRIDA]: {e}")
    
    return results



def inicializar_bm25():
    global _bm25
    global _bm25_docs

    if _bm25 is not None:
        return _bm25, _bm25_docs
    
    vector_store = carregar_vector_store()

    data = vector_store.get()

    textos = data.get("documents") or []
    metadatas = data.get("metadatas") or []

    if not textos:
        _bm25 = None
        _bm25_docs = []
        return _bm25, _bm25_docs

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

def reset_bm25():
    global _bm25, _bm25_docs
    
    _bm25 = None
    _bm25_docs = None


def _serializar_doc(doc, rank: int, score: float | None = None, include_content: bool = False):
    content = doc.page_content or ""
    item = {
        "rank": rank,
        "score": float(score) if score is not None else None,
        "metadata": dict(doc.metadata or {}),
        "preview": " ".join(content.split())[:500],
    }

    if include_content:
        item["content"] = content

    return item


def retrieval_debug(
    query: str,
    semantic_k: int = 10,
    bm25_k: int = 10,
    top_k: int = 5,
    include_content: bool = False,
) -> dict:
    vector_store = carregar_vector_store()

    semantic_docs = vector_store.max_marginal_relevance_search(
        query=query,
        k=semantic_k,
        fetch_k=max(semantic_k * 2, semantic_k),
    )

    bm25, bm25_docs = inicializar_bm25()
    bm25_results = []

    if bm25 is not None and bm25_docs:
        tokenized_query = query.lower().split()
        bm25_results = bm25.get_top_n(
            tokenized_query,
            bm25_docs,
            n=bm25_k,
        )

    combined_docs = semantic_docs + bm25_results
    unique_docs = list({
        doc.page_content: doc
        for doc in combined_docs
    }.values())

    if unique_docs:
        pairs = [
            (query, doc.page_content)
            for doc in unique_docs
        ]
        scores = reranker.predict(pairs) # type: ignore
        ranked = sorted(
            zip(unique_docs, scores),
            key=lambda x: x[1],
            reverse=True,
        )
    else:
        ranked = []

    MIN_RERANK_SCORE = -5.0

    reranked_docs = [
        (doc, score)
        for doc, score in ranked
        if score >= MIN_RERANK_SCORE
    ][:top_k]

    return {
        "query": query,
        "counts": {
            "semantic": len(semantic_docs),
            "bm25": len(bm25_results),
            "combined": len(combined_docs),
            "unique": len(unique_docs),
            "reranked": len(reranked_docs),
        },
        "semantic": [
            _serializar_doc(doc, rank=i + 1, include_content=include_content)
            for i, doc in enumerate(semantic_docs)
        ],
        "bm25": [
            _serializar_doc(doc, rank=i + 1, include_content=include_content)
            for i, doc in enumerate(bm25_results)
        ],
        "reranked": [
            _serializar_doc(doc, rank=i + 1, score=score, include_content=include_content)
            for i, (doc, score) in enumerate(reranked_docs)
        ],
    }

@measure_time("retrieve_docs")
def retrieve_docs(state: RAGState) -> dict:
    
    query = state.get("rewritten_query") or state["query"]
    
    reranked_docs = busca_hibrida_pgvector(query=query, limit=5)

    context = formatar_docs(reranked_docs)

    observe_docs(state["query"],query,reranked_docs)

    return {
        "context": context
    }

@measure_time("generate_answer")
def generate_answer(state: RAGState):
    prompt = ChatPromptTemplate.from_messages([
        ("system",
        """Você é um assistente que responde APENAS com base no contexto. Não invente nada!
        Caso não saiba a resposta com base no contexto, responda \"Não tenho informação sobre isso\"
         
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

graph = StateGraph(RAGState) #type: ignore

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
