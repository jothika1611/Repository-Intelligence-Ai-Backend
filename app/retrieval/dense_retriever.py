import logging
from typing import List, Dict, Any
from qdrant_client.http import models

from app.db.qdrant_client import qdrant_service, COLLECTION_NAME
from app.embeddings.dense import dense_embedding_model

logger = logging.getLogger(__name__)

class DenseRetriever:
    @staticmethod
    async def search(repository_id: str, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Executes a dense vector similarity search in Qdrant for a specific repository.
        """
        try:
            # 1. Generate query embedding
            # encode_batch expects an EmbeddingInput, but we can just use the raw sentence_transformer
            # model for a single query.
            vector = dense_embedding_model.model.encode(
                query,
                normalize_embeddings=True
            ).tolist()
            
            # 2. Query Qdrant with repository filter
            search_result = await qdrant_service.client.query_points(
                collection_name=COLLECTION_NAME,
                query=vector,
                limit=top_k,
                query_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="repository_id",
                            match=models.MatchValue(value=repository_id)
                        )
                    ]
                )
            )
            
            # 3. Format results
            formatted_results = []
            for hit in search_result.points:
                formatted_results.append({
                    "id": hit.id,
                    "score": hit.score,
                    "text": hit.payload.get("text", ""),
                    "source_type": hit.payload.get("source_type", ""),
                    "title": hit.payload.get("title", ""),
                    "url": hit.payload.get("url", ""),
                    "repository_id": hit.payload.get("repository_id", "")
                })
                
            return formatted_results
            
        except Exception as e:
            logger.error(f"Dense retrieval failed for repo {repository_id}: {e}", exc_info=True)
            raise

dense_retriever = DenseRetriever()
