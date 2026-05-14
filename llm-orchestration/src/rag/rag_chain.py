from typing import TypedDict, List
from rank_bm25 import BM25Okapi
from langchain_classic.retrievers.document_compressors import LLMChainExtractor
from langchain_classic.retrievers import ContextualCompressionRetriever
from sentence_transformers import CrossEncoder
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, BaseMessage, AIMessage
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from src.rag.vector_store import carregar_vector_store

load_dotenv()

_bm25 = None
_bm25_docs = None

llm = ChatGroq(model="llama-3.1-8b-instant")

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)



class RAGState(TypedDict, total=False):
    query: str
    context: str
    response: str
    filter: dict
    messages: List[BaseMessage]


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

def retrieve_docs(state: RAGState) -> dict:
    query = state["query"]
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

    compressor = LLMChainExtractor.from_llm(llm)

    compressed_docs = compressor.compress_documents(
        reranked_docs,
        query
    )
    # =====================================================
    # 6. FORMATAR CONTEXTO
    # =====================================================

    if not compressed_docs:
        compressed_docs = reranked_docs

    context = formatar_docs(compressed_docs)

    return {
        "context": context
    }

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

    return {
        "response": response,
        "messages": state.get("messages", []) + [
            HumanMessage(content=state["query"]),
            AIMessage(content=response)
        ]
    }

graph = StateGraph(RAGState)

graph.add_node("retrieve", retrieve_docs)
graph.add_node("generate", generate_answer)

graph.set_entry_point("retrieve")
graph.add_edge("retrieve", "generate")
graph.add_edge("generate", END)

memory = MemorySaver()
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