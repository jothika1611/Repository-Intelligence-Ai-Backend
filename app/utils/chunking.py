import hashlib
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.schemas.chunking import DocumentInput, ChunkOutput

class DocumentChunker:
    def __init__(self):
        # 1 token ~= 4 chars approximation. Target 300-500 tokens.
        # We will use 1600 characters as target size, with 200 characters overlap (approx 50 tokens overlap).
        self.chunk_size = 1600
        self.chunk_overlap = 200
        
        # Priority: Code blocks > Markdown Headers > Paragraphs > Sentences > Words
        self.separators = [
            "\n```\n",
            "\n# ",
            "\n## ",
            "\n### ",
            "\n\n",
            "\n",
            " ",
            ""
        ]
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
            keep_separator=True
        )

    def _generate_chunk_id(self, repository_id: str, text: str) -> str:
        # We use SHA-256 to create a deterministic hash ID to avoid duplicates
        # and ensure idempotent insertion for the same content.
        hasher = hashlib.sha256()
        hasher.update(repository_id.encode("utf-8"))
        hasher.update(text.encode("utf-8"))
        return hasher.hexdigest()

    def chunk_document(self, doc: DocumentInput) -> List[ChunkOutput]:
        if not doc.content or len(doc.content.strip()) == 0:
            return []
            
        content = doc.content.strip()
        
        # If the content is very small, we don't chunk it to preserve context
        if len(content) < 200:
            chunk_id = self._generate_chunk_id(doc.metadata.repository_id, content)
            return [
                ChunkOutput(
                    chunk_id=chunk_id,
                    text=content,
                    source_type=doc.source_type,
                    title=doc.title,
                    metadata=doc.metadata
                )
            ]

        # Use LangChain's recursive text splitter for everything else
        raw_chunks = self.text_splitter.split_text(content)
        
        chunk_outputs = []
        seen_hashes = set()
        
        for text in raw_chunks:
            chunk_text = text.strip()
            if not chunk_text:
                continue
                
            chunk_id = self._generate_chunk_id(doc.metadata.repository_id, chunk_text)
            
            # Deduplication: Avoid returning exactly identical chunks
            if chunk_id in seen_hashes:
                continue
            seen_hashes.add(chunk_id)
            
            chunk_outputs.append(
                ChunkOutput(
                    chunk_id=chunk_id,
                    text=chunk_text,
                    source_type=doc.source_type,
                    title=doc.title,
                    metadata=doc.metadata
                )
            )
            
        return chunk_outputs

document_chunker = DocumentChunker()
