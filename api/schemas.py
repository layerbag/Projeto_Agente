from pydantic import BaseModel, Field

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    session_id: str | None = None

class ChatResponse(BaseModel):
    session_id: str
    response: str

class IndexDoc(BaseModel):
    type: str
    url: str

class DeleteDocumentRequest(BaseModel):
    source: str = Field(..., min_length=1)

class RetrievalRequest(BaseModel):
    query: str = Field(..., min_length=1)
    semantic_k: int = Field(default=10, ge=1, le=30)
    bm25_k: int = Field(default=10, ge=1, le=30)
    top_k: int = Field(default=5, ge=1, le=20)
    include_content: bool = False

class AgentChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    session_id: str | None = None
    
class AgentChatResponse(BaseModel):
    session_id: str
    response: str
