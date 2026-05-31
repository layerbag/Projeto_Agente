import numpy as np
import yt_dlp
from transformers import AutoTokenizer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_core.documents import Document
from src.rag.vector_store import get_embeddings
from langchain_experimental.text_splitter import SemanticChunker
import re
# pyrefly: ignore [missing-import]
from src.utils.project_utils import load_pdf, yt2doc

VIDEO_URL = "https://www.youtube.com/watch?v=JsD1ewgzLJc"

try:
    _BGE_TOKENIZER = AutoTokenizer.from_pretrained("BAAI/bge-m3")
except Exception as e:
    print(f"Erro ao carregar o tokenizer: {e}")
    _BGE_TOKENIZER = None

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
def dividir_em_chunks(docs: list[Document] = None) -> list[Document]: # type: ignore

    if not docs:
        print("Nenhum documento para dividir em chunks.")
        return []

    chunks: list[Document] = []

    for doc in docs:
        chunks.extend(semantic_chunking(doc))

    print(f"Total de chunks gerados: {len(chunks)}")
    return chunks

def _contar_tokens(texto: str) -> int:
    if _BGE_TOKENIZER is None:
        return len(texto) // 4
    return len(_BGE_TOKENIZER.tokenize(texto))

def semantic_chunking(doc: Document, threshold_percentile: float = 85.0, max_chunk_tokens = 2048) -> list[Document]:
    """Realiza a separação de chunkings com base na distância de cosseno dos documentos"""

    texto = doc.page_content
    metadata = doc.metadata

    # dividir o texto em sentenças criadas ao separar frases que terminam com ., ?, ! antes de um espaço.
    # Ex: Olá! Eu sou o programador.
    # Sentenças: "Olá!", "Eu sou o programador."
    sentences = re.split(r'(?<=[.?!])\s+', texto.strip())
    #filtrar sentenças curtas
    sentences = [s for s in sentences if len(s.strip()) > 10]

    
    if len(sentences) <= 1:
        return [doc]

    embeddings = get_embeddings().embed_documents(sentences)

    embeddings = [np.array(e) for e in embeddings]

    similaridades = []

    for i in range(len(embeddings) - 1):
        v1 = embeddings[i]
        v2 = embeddings[i+1]

        # distancia de cosseno
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)

        if norm_v1 > 0 and norm_v2 > 0:
            cos_sim = np.dot(v1,v2) / (norm_v1 * norm_v2)
        else:
            cos_sim = 0
        
        similaridades.append(cos_sim)

    # encontrar o threshold de distância baseada no percentil desejado
    # Quanto menor a similaridade, maior a distância semântica

    distancias = [1.0 - sim for sim in similaridades]
    limiar = np.percentile(distancias,threshold_percentile)

    # agrupar as sentenças em chunks usando o limiar

    chunks = []
    chunk_atual = [sentences[0]]

    for i in range(len(sentences) - 1):
        distancia = distancias[i]

        if distancia > limiar:
            chunks.append(" ".join(chunk_atual))
            chunk_atual = [sentences[i+1]]

        else:
            chunk_atual.append(sentences[i+1])
    
    if chunk_atual:
        chunks.append(" ".join(chunk_atual))

    # Converter os blocos de texto de volta em instâncias de "Document"

    documentos_chunks = []

    for i, trecho in enumerate(chunks):
        new_meta = metadata.copy()
        new_meta["chunk_index"] = i
        documentos_chunks.append(Document(page_content=trecho, metadata=new_meta))
    

    token_splitter = RecursiveCharacterTextSplitter(
        chunk_size= max_chunk_tokens,
        chunk_overlap=int(2048*0.1),
        length_function=_contar_tokens,
        separators=["\n\n","\n"," ",""]
    )

    final_chunks = []
    for chunk in documentos_chunks:
        if _contar_tokens(chunk.page_content) > max_chunk_tokens:
            sub_chunks = token_splitter.split_documents([chunk])
            
            for j, sub_doc in enumerate(sub_chunks):
                sub_doc.metadata["sub_chunk_index"] = j
                final_chunks.append(sub_doc)
        else:
            final_chunks.append(chunk)

    return final_chunks

# main para teste
if __name__ == "__main__":
    video_id, title = extrair_informacoes(VIDEO_URL)
    transcricao = extrair_transcricao(video_id)
    metadata = {"type": "YouTube", "video_id": video_id, "title": title, "source": VIDEO_URL} # type: ignore
    pdfList = load_pdf("./Currículo_Gabriel_Martins.pdf")
    # pyrefly: ignore [unexpected-keyword]
    chunks = dividir_em_chunks(pdf_docs=pdfList, yt_docs=yt2doc(transcricao, metadata))
    
