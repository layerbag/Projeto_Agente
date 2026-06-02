import os
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from sqlalchemy import create_engine
from langchain_postgres.vectorstores import PGVector

load_dotenv()
_embeddings = None

CONNECTION_STRING = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg://postgres:my_secure_password@localhost:5432/orchestration_db"
)

_engine = create_engine(CONNECTION_STRING)

COLLECTION_NAME = "documents_collection"

# configura o modelo de embeddings
def get_embeddings():
    """Configura e retorna o modelo de embeddings."""
    global _embeddings

    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-m3",
            model_kwargs={"device": "cuda"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings

def get_engine():
    """retorna a engine de conexão com o PGVector"""
    global _engine
    return _engine

# Funções para criar, carregar e adicionar ao vector store do Chroma
def criar_vector_store(chunks): # type: ignore
    """Cria ou carrega o vector store do Chroma e insere os chunks."""
    print("Criando e indexando dados no PostgreSQL (pgvector)")
    db = PGVector.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        collection_name=COLLECTION_NAME,
        connection=_engine,
        use_jsonb=True
    )
    return db

# def add_vector_store(chunks): # type: ignore
#     if not chunks or len(chunks) == 0:
#         print("Nenhum chunk para adicionar ao vector store.")
#         return
#     """Adiciona novos chunks ao vector store existente."""
#     print("Adicionando novos chunks ao vector store do Chroma...")
#     db = carregar_vector_store()
#     db.add_documents(chunks)  # type: ignore
#     print(f"{db._collection.count()} chunks agora indexados em {CHROMA_DIR}")  # type: ignore

def carregar_vector_store():
    """Carrega o vector store do Chroma existente."""
    return PGVector(
        connection=_engine,
        embeddings=get_embeddings(),
        collection_name=COLLECTION_NAME,
        use_jsonb=True
    )

# def busca_semantica(query:str, k:int = 3):
#     """Busca os k chunks mais relevantes para a query usando busca semântica."""
#     vector_store = carregar_vector_store()
#     resultados = vector_store.similarity_search(query, k=k)
#     for i, doc in enumerate(resultados):
#         print(f"\n Chunk {i+1}: \n{doc.page_content}")
#     return resultados
