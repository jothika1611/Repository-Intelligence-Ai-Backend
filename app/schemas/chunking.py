from pydantic import BaseModel, Field
from typing import List, Optional

class DocumentMetadata(BaseModel):
    url: str
    repository_id: str

class DocumentInput(BaseModel):
    source_type: str = Field(..., description="readme | documentation | issue | pull_request")
    title: str
    content: str
    metadata: DocumentMetadata

class ChunkOutput(BaseModel):
    chunk_id: str
    text: str
    source_type: str
    title: str
    metadata: DocumentMetadata
