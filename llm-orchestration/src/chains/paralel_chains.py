from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(model="llama-3.1-8b-instant")

# 3 chains com perpspectivas diferentes

chain_pros = (
    ChatPromptTemplate.from_template("Liste 3 vantagens de {tema}. Somente liste as vantagens em lista, sem explicar o {tema}") | llm | StrOutputParser()
)

chain_cons = (
    ChatPromptTemplate.from_template("Liste 3 desvantagens de {tema}. Somente liste as desvantagens em lista, sem explicar o {tema}") | llm | StrOutputParser()
)

chain_resumo = (
    ChatPromptTemplate.from_template("Faça um resumo sobre {tema}. Responda em Português.") | llm | StrOutputParser()
)

# Chain paralela
analise_completa = RunnableParallel(
    vantagens = chain_pros,
    desvantagens = chain_cons,
    resumo = chain_resumo,
)

resultado = analise_completa.invoke({"tema": "inteligência artificial"})

print("Resumo:", resultado["resumo"])
print("Vantagens:", resultado["vantagens"])
print("Desvantagens:", resultado["desvantagens"])