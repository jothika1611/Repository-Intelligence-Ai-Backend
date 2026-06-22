from pydantic import BaseModel
from typing import List
from app.schemas.chunking import ChunkOutput, DocumentMetadata

class EmbeddedChunk(BaseModel):
    chunk_id: str
    vector: List[float]
    text: str
    source_type: str
    title: str
    metadata: DocumentMetadata

class EmbeddingInput(BaseModel):
    chunks: List[ChunkOutput]

class EmbeddingOutput(BaseModel):
    embeddings: List[EmbeddedChunk]
