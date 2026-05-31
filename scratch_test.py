import traceback
from src.rag.rag_chain import busca_hibrida_pgvector

print("Iniciando teste de busca híbrida...")
try:
    results = busca_hibrida_pgvector("O que é Ordem Paranormal?")
    print(f"Sucesso! {len(results)} documentos recuperados.")
except Exception as e:
    print("ERRO DETECTADO:")
    traceback.print_exc()
