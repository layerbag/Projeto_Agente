from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(model="llama-3.1-8b-instant")

# Definição do prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "Você é um especialista em {area}. Responda de forma clara e concisa"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{pergunta}")
])

# Montagem da chain
chain = prompt | llm | StrOutputParser()

store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="pergunta",
    history_messages_key="history",
)

config = {"configurable": {"session_id": "sessao-123"}}

r1 = chain_with_history.invoke({
    "area": "Inteligência Artificial", 
    "pergunta": "Meu nome é Gabriel."}, config=config)
print(f"Bot: {r1}")

r2 = chain_with_history.invoke({
    "area": "Inteligência Artificial", 
    "pergunta": "Qual é o meu nome?"}, config=config)
print(f"Bot: {r2}")