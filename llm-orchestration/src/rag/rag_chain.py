from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from dotenv import load_dotenv
from src.rag.vector_store import carregar_vector_store

load_dotenv()

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
def recuperar_docs(inputs): # type: ignore
    query = inputs["query"]
    filter = inputs.get("filter", {})

    if filter:
        search_kwargs = {"k": 5, "filter": filter}
    else:
        search_kwargs = {"k": 5}

    retriever = carregar_vector_store().as_retriever(search_type="similarity", search_kwargs=search_kwargs)  # type: ignore
    docs = retriever.invoke(query)  # type: ignore

    return docs

# criar a cadeia RAG usando o ChatGroq e o vector store
def criar_rag_chain():
    """Cria a cadeia RAG usando o ChatGroq e o vector store."""
    llm = ChatGroq(model="llama-3.1-8b-instant")  # type: ignore

    # Prompt template para a cadeia RAG
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", """
         Você é um assistente que responde perguntas com base em informações dos documentos. Use os documentos fornecidos para responder à pergunta do usuário. Use APENAS as informações dos documentos para responder, e não invente nada que não esteja lá.
         contexto da transcrição: {context}
         """),
        ("human", "{query}")
    ])

    chain = (
        {
            "context": RunnableLambda(recuperar_docs) | formatar_docs, # type: ignore
            "query": RunnablePassthrough(),
        }
        | prompt_template
        | llm
        | StrOutputParser()
    )

    return chain

# main para teste
if __name__ == "__main__":
    rag_chain = criar_rag_chain()
    perguntas = [
        "O que acontece no vídeo?",
        "Qual é a reação do apresentador?",
        "A quem ele se refere quando fala 'oque que você fez?'?"
        "Sobre qual jogo é o vídeo?"
    ]

    for pergunta in perguntas:
        print(f"\nPergunta: {pergunta}")
        resposta = rag_chain.invoke({"query": pergunta})
        print("\nresposta:\n", resposta)