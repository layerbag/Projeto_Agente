import platform
import unicodedata
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from src.rag.vector_store import carregar_vector_store

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


def load_pdf(file_path: str):
    """Carrega um PDF e retorna seu conteúdo como Document"""
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    # Adiciona metadados e formata o conteúdo de cada página
    for doc in documents:
        print("tamanho: ", len(doc.page_content))
        print(repr(doc.page_content[:100]))
        doc.metadata["type"] = "pdf"
        doc.metadata["title"] = file_path.split("/")[-1]  # type: ignore
        doc.page_content = f"""
        Título: {file_path.split('/')[-1]}

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
    vector_store = carregar_vector_store()
    source_name = normalizar_caminho(source)

    data = vector_store.get(include=["metadatas"])

    ids_to_delete = []

    for chunk_id, metadata in zip(data.get("ids", []), data.get("metadatas", [])):
        metadata = metadata or {}

        metadata_source = str(metadata.get("source", ""))
        metadata_url = str(metadata.get("url", ""))
        metadata_title = str(metadata.get("title", ""))
        metadata_source_name = normalizar_caminho(metadata_source)
    
        matches_source = metadata_source == source
        matches_url = metadata_url == source
        matches_title = metadata_title == source
        matches_source_name = metadata_source_name == source_name

        if matches_source or matches_url or matches_title or matches_source_name:
            ids_to_delete.append(chunk_id)

    if not ids_to_delete:
        return {
            "deleted": False,
            "deleted_chunks": 0,
            "message": "Nenhum documento encontrado para deletar.",
        }

    vector_store.delete(ids=ids_to_delete)

    return {
        "deleted": True,
        "deleted_chunks": len(ids_to_delete),
        "message": "Documento removido do Chroma.",
    }