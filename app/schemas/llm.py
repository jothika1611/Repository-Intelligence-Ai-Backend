from pydantic import BaseModel
from typing import List

class LLMRequest(BaseModel):
    prompt: str
    context: List[str]
    repository_id: str

class LLMResponse(BaseModel):
    response: str
    provider: str
