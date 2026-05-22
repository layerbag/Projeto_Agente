import yt_dlp
from langchain_text_splitters import RecursiveCharacterTextSplitter
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_core.documents import Document
from src.utils.project_utils import load_pdf, yt2doc

VIDEO_URL = "https://www.youtube.com/watch?v=JsD1ewgzLJc"

# extrair informações do vídeo
def extrair_informacoes(url: str):
    """extrai as informações e retorna a tupla (videoId, Titulo)"""
    with yt_dlp.YoutubeDL({"quiet": True}) as ydl:
        info = ydl.extract_info(url, download=False)
    
    video_id = info.get("id")
    title = info.get("title", "Sem título")

    print(f"Informações extraídas: video_id={video_id}, title={title}")
    return video_id, title

# Extrair a transcrição do vídeo usando a API do YouTube
def extrair_transcricao(video_id: str):
    api = YouTubeTranscriptApi()
    transcript = api.fetch(video_id, languages=['pt', 'en']) # type: ignore
    texto = " ".join([item.text for item in transcript]) # type: ignore
    
    return texto

# Dividir os documentos em chunks menores para indexação
def dividir_em_chunks(docs: list[Document] = None): # type: ignore

    if not docs:
        print("Nenhum documento para dividir em chunks.")
        return []

    text_splitter_yt = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150, separators=["\n\n", "\n"," ", ""] )
    text_splitter_pdf = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200, separators=["\n\n", "\n"," ", ""] )

    pdf_docs = []
    yt_docs = []

    if not docs:
        print("Nenhum documento para dividir em chunks.")
        return []

    for doc in docs:
        if doc.metadata["type"] == "YouTube":
            yt_docs.append(doc)
        else:
            pdf_docs.append(doc)
    
    # splitting dos documentos em chunks menores
    chunks = []
    
    if pdf_docs:
        chunks.extend(text_splitter_pdf.split_documents(pdf_docs))

    if yt_docs:
        chunks.extend(text_splitter_yt.split_documents(yt_docs))

    # Filtrar chunks muito pequenos ou que contenham apenas metadados
    chunks = [
        chunk for chunk in chunks
        if len(chunk.page_content.strip()) > 100
    ]


    print(f"Total de chunks gerados: {len(chunks)}")
    return chunks

# main para teste
if __name__ == "__main__":
    video_id, title = extrair_informacoes(VIDEO_URL)
    transcricao = extrair_transcricao(video_id)
    metadata = {"type": "YouTube", "video_id": video_id, "title": title, "source": VIDEO_URL} # type: ignore
    pdfList = load_pdf("./Currículo_Gabriel_Martins.pdf")
    chunks = dividir_em_chunks(pdf_docs=pdfList, yt_docs=yt2doc(transcricao, metadata))
    