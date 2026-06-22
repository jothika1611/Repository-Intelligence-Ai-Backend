import logging
from typing import List, Dict, Any
from qdrant_client.http import models
from rank_bm25 import BM25Okapi

from app.db.qdrant_client import qdrant_service, COLLECTION_NAME

logger = logging.getLogger(__name__)

class SparseRetriever:
    @staticmethod
    async def _fetch_all_chunks(repository_id: str) -> List[Dict[str, Any]]:
        """
        Fetches all chunks for a repository from Qdrant to build an in-memory BM25 index.
        """
        chunks = []
        offset = None
        while True:
            result, offset = await qdrant_service.client.scroll(
                collection_name=COLLECTION_NAME,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="repository_id",
                            match=models.MatchValue(value=repository_id)
                        )
                    ]
                ),
                limit=1000,
                offset=offset,
                with_payload=True,
                with_vectors=False
            )
            for point in result:
                chunks.append({
                    "id": point.id,
                    "text": point.payload.get("text", ""),
                    "source_type": point.payload.get("source_type", ""),
                    "title": point.payload.get("title", ""),
                    "url": point.payload.get("url", ""),
                    "repository_id": point.payload.get("repository_id", "")
                })
            if offset is None:
                break
        return chunks

    @staticmethod
    async def search(repository_id: str, query: str, top_k: int = 50) -> List[Dict[str, Any]]:
        """
        Executes a sparse BM25 search dynamically for a specific repository.
        """
        try:
            chunks = await SparseRetriever._fetch_all_chunks(repository_id)
            if not chunks:
                return []
                
            # Simple tokenization by whitespace for BM25
            tokenized_corpus = [chunk["text"].lower().split() for chunk in chunks]
            bm25 = BM25Okapi(tokenized_corpus)
            
            tokenized_query = query.lower().split()
            scores = bm25.get_scores(tokenized_query)
            
            # Filter and sort by score descending
            scored_chunks = [(score, chunk) for score, chunk in zip(scores, chunks) if score > 0]
            scored_chunks.sort(key=lambda x: x[0], reverse=True)
            
            formatted_results = []
            for score, chunk in scored_chunks[:top_k]:
                chunk_data = chunk.copy()
                chunk_data["score"] = score
                formatted_results.append(chunk_data)
                
            return formatted_results
            
        except Exception as e:
            logger.error(f"Sparse retrieval failed for repo {repository_id}: {e}", exc_info=True)
            raise

sparse_retriever = SparseRetriever()
