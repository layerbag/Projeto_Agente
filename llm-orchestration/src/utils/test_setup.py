"""Script de validação do ambiente"""
import os
from dotenv import load_dotenv

load_dotenv()

def test_groq():
    from langchain_groq import ChatGroq
    llm = ChatGroq(model="llama-3.1-8b-instant")

    resp = llm.invoke("What is the capital of France?")
    print(f"✅ Resposta do Groq: {resp.content}")

def test_gemini():
    from langchain_google_genai import ChatGoogleGenerativeAI
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

    resp = llm.invoke("What is the capital of France?")
    print(f"✅ Resposta do Gemini: {resp.content}")

def test_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings
    emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vec = emb.embed_query("Hello world")
    print(f"✅ Vetor de {len(vec)} dimensões, embedding: {vec[:5]}...")

def test_chroma():
    import chromadb
    client = chromadb.PersistentClient(path=os.getenv("CHROMA_PERSIST_DIR","./data/chroma_db"))
    print(f"✅ ChromaDB: {len(client.list_collections())} coleções")

if __name__ == "__main__":
    # print("🔍 Testando integração com Groq...")
    # test_groq()
    print("\n🔍 Testando integração com Gemini...")
    test_gemini()
    print("\n🔍 Testando integração com HuggingFace Embeddings...")
    test_embeddings()
    print("\n🔍 Testando integração com ChromaDB...")
    test_chroma()
    print("\n✅ Todos os testes concluídos com sucesso!")