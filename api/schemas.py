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