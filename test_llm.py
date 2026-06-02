from src.rag.rag_chain import llm
try:
    print("Calling llm...")
    res = llm.invoke("Hello")
    print("Success:", res)
except Exception as e:
    print("Exception caught:", e)
