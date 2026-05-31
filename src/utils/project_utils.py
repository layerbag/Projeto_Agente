import platform
import unicodedata
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from sqlalchemy import text
# pyrefly: ignore [missing-import]
from src.rag.vector_store import carregar_vector_store, get_engine

def normalizar_caminho(caminho: str) -> str:
    """Normaliza o caminho do documento para evitar duplicações no vector store.
    - WSL converte C:\\ para /mnt/c/
    - Windows C:\\...
    - Linux C\\ vira C/
    """
    sistema = platform.system()

    if caminho.__contains__(".com"):
        return caminho

    if "microsoft" in platform.uname().release.lower():
        # Converte C:\... para /mnt/c/...
        if ":" in caminho:
            drive, resto = caminho.split(":", 1)
            caminho = f"/mnt/{drive.lower()}{resto.replace('\\', '/')}"
        return caminho
    
    if sistema == "Windows":
        return caminho
    
    if sistema == "Linux":
        return caminho.replace("\\", "/")
    
    return caminho


def load_pdf(file_path: str, Title: str):
    """Carrega um PDF e retorna seu conteúdo como Document"""
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    # Adiciona metadados e formata o conteúdo de cada página
    for doc in documents:
        doc.metadata["type"] = "pdf"
        doc.metadata["title"] = Title  # type: ignore
        doc.page_content = f"""
        Título: {Title}

        Conteúdo:
        {doc.page_content}
        """  # type: ignore

    print(f"PDF carregado: {len(documents)} páginas extraídas.")
    print(f"Exemplo de metadados da primeira página: {documents[0].metadata}")  # type: ignore
    return documents

# Converter a transcrição do YouTube em um documento formatado
def yt2doc (texto:str, metadata: dict): # type: ignore
    """Converte a transcrição do YouTube em um documento formatado."""
    page_content = f"""
    Título: {metadata.get("title", "Sem título")}

    Conteúdo: 
    {texto}
    """
    return [Document(page_content=page_content, metadata=metadata)] # type: ignore

def delete_indexed_document(source: str) -> dict:
    engine = get_engine()

    try:
        with engine.begin() as conn:
            result = conn.execute(
                text("""
                    DELETE FROM langchain_pg_embedding
                    WHERE cmetadata->>'title' = :source

                """),
                {"source" : source}
            )

            deleted_count = result.rowcount

            if deleted_count > 0:
                return {
                "deleted": True,
                "message": f"{deleted_count} registros deletados com sucesso."
                }
            else:
                return {
                    "deleted": False,
                    "message": "Nenhum registro encontrado para deletar."
                }

    except Exception as e:
        return {
            "deleted": False,
            "message": f"Erro ao deletar documento: {str(e)}"
        }
