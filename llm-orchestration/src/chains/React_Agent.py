from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(model="llama-3.1-8b-instant")

# tool 1 - busca na web
search = DuckDuckGoSearchRun()

# tool 2 - calculadora customizada
@tool
def calculadora (expressao: str) -> str:
    """Calcula expressões matemáticas simples. Exemplos: 2 + 2, 10 * 5.5, 100 / 4"""
    try:
        resultado = eval(expressao)
        return f'Resultado: {resultado}'
    except Exception as e:
        return f'Erro na expressão: {e}'

tools = [search, calculadora]

agent = create_agent(model=llm, tools=tools, system_prompt="""Você é um assistente que sempre responde em português e traduz para inglês.""")

resultado = agent.invoke(
    {"messages":[{"role": "user", "content": "Qual é a capital da França e quanto é 15 * 3?"}]}
)

print("\nResposta do agente:", resultado["messages"][-1].content)