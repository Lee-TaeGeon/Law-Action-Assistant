from typing import List
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(min_length=1)
    session_id: str = "default"


class LawSource(BaseModel):
    law_name: str
    article: str | None = None
    content: str


class ChatResponse(BaseModel):
    category: str
    answer: str
    sources: List[LawSource]