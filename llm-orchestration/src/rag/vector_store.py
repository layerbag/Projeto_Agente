import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

load_dotenv()
_embeddings = None

CHROMA_DIR = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")
COLLECTION = "youtube_videos"

# configura o modelo de embeddings
def get_embeddings():
    """Configura e retorna o modelo de embeddings."""
    global _embeddings

    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )
    return _embeddings

# Funções para criar, carregar e adicionar ao vector store do Chroma
def criar_vector_store(chunks): # type: ignore
    """Cria ou carrega o vector store do Chroma e insere os chunks."""
    print("Criando ou carregando o vector store do Chroma...")
    db = Chroma.from_documents( # type: ignore
        documents=chunks,
        embedding=get_embeddings(),
        collection_name=COLLECTION,
        persist_directory=CHROMA_DIR
    )
    print(f"{db._collection.count()} chunks indexados em {CHROMA_DIR}")  # type: ignore
    return db

def add_vector_store(chunks): # type: ignore
    if not chunks or len(chunks) == 0:
        print("Nenhum chunk para adicionar ao vector store.")
        return
    """Adiciona novos chunks ao vector store existente."""
    print("Adicionando novos chunks ao vector store do Chroma...")
    db = carregar_vector_store()
    db.add_documents(chunks)  # type: ignore
    print(f"{db._collection.count()} chunks agora indexados em {CHROMA_DIR}")  # type: ignore

def carregar_vector_store():
    """Carrega o vector store do Chroma existente."""
    return Chroma(
        persist_directory=CHROMA_DIR,
        collection_name=COLLECTION,
        embedding_function=get_embeddings()
    )

def busca_semantica(query:str, k:int = 3):
    """Busca os k chunks mais relevantes para a query usando busca semântica."""
    vector_store = carregar_vector_store()
    resultados = vector_store.similarity_search(query, k=k)
    for i, doc in enumerate(resultados):
        print(f"\n Chunk {i+1}: \n{doc.page_content}")
    return resultados