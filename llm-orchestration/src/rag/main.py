import os
import platform
from dotenv import load_dotenv
from src.rag.rag_chain import app
from src.rag.vector_store import carregar_vector_store
from src.rag.ingestao import extrair_informacoes, extrair_transcricao, dividir_em_chunks, load_pdf, yt2doc
from src.mlops.observability import rag_com_rastreamento

load_dotenv()

CHROMA_DIR = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")

def normalizar_caminho(caminho: str) -> str:
    """Normaliza o caminho do documento para evitar duplicações no vector store."""
    sistema = platform.system()

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

# indexar os documentos se necessário e criar a cadeia RAG
def indexar_se_necessario(pdf_docs: list[dict[str, str]], yt_docs: list[dict[str, str]]):
    """Indexa os documentos se o documento ainda não foi indexado, verificando o vector store do Chroma."""
    yt_loaded = []
    pdf_loaded = []
    vector_store = carregar_vector_store()

    # adiciona os documentos PDF e de vídeo ao vector store se ainda não estiverem indexados, evitando duplicações
    for doc in pdf_docs:
        resultados = vector_store.get(
            where={"source": doc["caminho"]}  # type: ignore
        )
        print(resultados)
        if len(resultados["ids"]) == 0:
            pdf_loaded.extend(load_pdf(doc["caminho"]))
        else:
            print("Documento PDF já indexado:", doc["caminho"])

    # adiciona os documentos de vídeo ao vector store se ainda não estiverem indexados, evitando duplicações
    for doc in yt_docs:
        resultados = vector_store.get(
            where={"source": doc["caminho"]}  # type: ignore
        )

        if len(resultados["ids"]) == 0:
            video_id, title = extrair_informacoes(doc["caminho"])
            texto = extrair_transcricao(video_id)
            metadata = {"video_id": video_id, "title": title, "source": doc["caminho"]}
            yt_loaded.extend(yt2doc(texto, metadata))
        else:
            print("Documento de vídeo já indexado:", doc["caminho"])

    # divide os documentos em chunks e adiciona ao vector store
    chunks = dividir_em_chunks(yt_docs = yt_loaded, pdf_docs = pdf_loaded)
    if chunks:
        vector_store.add_documents(chunks)

    print("Indexação concluída. Documentos carregados:", len(pdf_loaded) + len(yt_loaded))
    return vector_store

# main para perguntas e respostas usando a cadeia RAG
def main():
    while True:
        operacao = input("Digite \"I\" para inserir novos documento ou \"C\" para iniciar o chat: ").strip().lower()

        #inserir novos documentos
        if operacao == "i":
            pdf_docs = []
            yt_docs = []
            
            print("Digite \"tipo(pdf ou vídeo) caminho\" para indexar um documento ou \"fim\" para concluir: ")
            # batch de novos documentos para indexar (ex: "pdf /caminho/para/arquivo.pdf" ou "vídeo https://www.youtube.com/watch?v=exemplo")
            while True:
                linha = input().strip()

                if linha.lower() == "fim":
                    break

                partes = linha.split(maxsplit=1)

                if len(partes) != 2:
                    print("Entrada inválida. Por favor, digite no formato \"tipo caminho\" ou \"fim\" para concluir.")
                    continue

                tipo_doc, caminho = partes

                if tipo_doc.lower() == "pdf":
                    caminho = normalizar_caminho(caminho)
                    pdf_docs.append({"tipo": "pdf", "caminho": caminho})
                elif tipo_doc.lower() == "vídeo" or tipo_doc.lower() == "video":
                    yt_docs.append({"tipo": "YouTube", "caminho": caminho})
                else:
                    print("Tipo de documento inválido. Por favor, digite \"pdf\" ou \"vídeo\" seguido do caminho.")
                    continue

            indexar_se_necessario(pdf_docs, yt_docs)    
            

        elif operacao == "c":
            config = {"configurable": {"thread_id": "sessao-123"}}

            while True:
                pergunta = input("\nDigite sua pergunta (ou \"sair\" para encerrar): ").strip()

                if pergunta.lower() in {"sair", "exit", "quit"}:
                    print("Encerrando o programa.")
                    break
                if not pergunta:
                    continue
        
                resposta = app.invoke(
                    {"query": pergunta},
                    config = config
                )
                print(f"\nResposta: {resposta['response']}")
        
        elif operacao == "sair":
            print("Encerrando o programa.")
            break
        else:
            print("Operação inválida. Por favor, digite \"I\" para inserir documentos ou \"C\" para iniciar o chat.")
        

if __name__ == "__main__":
    carregar_vector_store()
    main()

    # indexar_se_necessario()
    # vs = carregar_vector_store()

    # docs = vs.similarity_search_with_score("Gabriel Martins", k=5, filter={"type": "pdf"})  # type: ignore

    # for doc, score in docs:
    #     print("\nSCORE:", score)
    #     print("METADATA:", doc.metadata)
    #     print("CONTEÚDO:", doc.page_content[:300])