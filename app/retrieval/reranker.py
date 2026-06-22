import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class CrossEncoderReranker:
    def __init__(self):
        try:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        except Exception as e:
            logger.error(f"Failed to load CrossEncoder model: {e}", exc_info=True)
            raise

    def rerank(self, query: str, results: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Reranks a list of candidate results against the query using a cross-encoder model.
        """
        if not results:
            return []
            
        # Build pairs
        pairs = [[query, res.get("text", "")] for res in results]
        
        # Predict scores in batch
        scores = self.model.predict(pairs)
        
        # Update results with new scores
        for idx, score in enumerate(scores):
            results[idx]["cross_encoder_score"] = float(score)
            
        # Sort descending
        results.sort(key=lambda x: x["cross_encoder_score"], reverse=True)
        
        return results[:top_k]

cross_encoder_reranker = CrossEncoderReranker()
