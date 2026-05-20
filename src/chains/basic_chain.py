from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

# Configuração do LLM
llm = ChatGroq(model="llama-3.1-8b-instant")

# Definição do prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "Você é um especialista em {area}. Responda de forma clara e concisa"),
    ("human", "{pergunta}")
])

# Montagem da chain
chain = prompt | llm | StrOutputParser()

# Execução da chain
for chunk in chain.stream({
    "area": "Inteligência Artificial",
    "pergunta": "Oque é RAG?",
}):
    print(chunk, end="",flush=True)