from typing import List
from pydantic import BaseModel


# Pydantic model for RAG response
class AnswerResponse(BaseModel):
    answer: str
    sources: List[str]
