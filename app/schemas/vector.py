from enum import Enum
from pydantic import BaseModel, Field

class SourceType(str, Enum):
    readme = "readme"
    docs = "docs"
    issue = "issue"
    pull_request = "pull_request"

class ChunkMetadata(BaseModel):
    repository_id: str = Field(..., description="Format: owner/repo")
    source_type: SourceType
    source_id: str = Field(..., description="Unique identifier for the source (e.g. issue number or file path)")
    title: str = Field(..., description="Title of the chunk's source document")
    url: str = Field(..., description="GitHub URL pointing to the source document")
    content: str = Field(..., description="The raw textual content of the chunk")
