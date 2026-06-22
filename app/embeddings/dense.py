import logging
import re
from typing import List
from sentence_transformers import SentenceTransformer

from app.schemas.config import settings
from app.schemas.chunking import ChunkOutput
from app.schemas.embeddings import EmbeddedChunk, EmbeddingInput, EmbeddingOutput

logger = logging.getLogger(__name__)

class DenseEmbeddingModel:
    def __init__(self):
        # Initialize the embedding model as a singleton to avoid reloading weights
        # We use BAAI/bge-small-en-v1.5 as dictated by the MVP constraints
        logger.info(f"Loading dense embedding model: {settings.embedding_model}")
        try:
            self.model = SentenceTransformer(settings.embedding_model)
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}", exc_info=True)
            raise
            
        self.vector_size = 384  # Expected for bge-small-en-v1.5

    def clean_text(self, text: str) -> str:
        """
        Cleans excessive whitespace safely.
        Avoids modifying anything that looks like a code block.
        """
        # If it contains a code block, skip aggressive cleaning to preserve indentation
        if "```" in text:
            return text.strip()
            
        # Replace multiple spaces with a single space
        text = re.sub(r'[ \t]+', ' ', text)
        # Replace 3+ newlines with exactly 2 newlines (paragraph separator)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text.strip()

    def encode_batch(self, input_data: EmbeddingInput, batch_size: int = 32) -> EmbeddingOutput:
        """
        Converts a list of ChunkOutput objects into a list of EmbeddedChunk objects.
        Uses SentenceTransformers batched encoding.
        """
        chunks = input_data.chunks
        if not chunks:
            return EmbeddingOutput(embeddings=[])
            
        # Extract and clean texts
        texts = [self.clean_text(chunk.text) for chunk in chunks]
        
        # Generate embeddings
        # We explicitly normalize embeddings for Cosine similarity
        try:
            vectors = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=False,
                normalize_embeddings=True
            )
        except Exception as e:
            logger.error(f"Failed to encode text batch: {e}", exc_info=True)
            raise
            
        # Map back to schemas
        embedded_chunks = []
        for chunk, vector in zip(chunks, vectors):
            embedded_chunks.append(
                EmbeddedChunk(
                    chunk_id=chunk.chunk_id,
                    vector=vector.tolist(),
                    text=chunk.text,
                    source_type=chunk.source_type,
                    title=chunk.title,
                    metadata=chunk.metadata
                )
            )
            
        return EmbeddingOutput(embeddings=embedded_chunks)

dense_embedding_model = DenseEmbeddingModel()
