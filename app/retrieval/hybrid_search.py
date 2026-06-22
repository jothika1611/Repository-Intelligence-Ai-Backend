import logging
import asyncio
from typing import List, Dict, Any

from app.retrieval.dense_retriever import dense_retriever
from app.retrieval.sparse_retriever import sparse_retriever
from app.retrieval.reranker import cross_encoder_reranker

logger = logging.getLogger(__name__)

class HybridRetriever:
    @staticmethod
    def rrf_fusion(dense_results: List[Dict[str, Any]], sparse_results: List[Dict[str, Any]], k: int = 60) -> List[Dict[str, Any]]:
        """
        Merges dense and sparse results using Reciprocal Rank Fusion (RRF).
        """
        fused_scores = {}
        chunk_map = {}
        
        # Helper to process lists
        def process_list(results: List[Dict[str, Any]]):
            for rank, hit in enumerate(results):
                chunk_id = hit.get("id")
                if not chunk_id:
                    continue
                    
                if chunk_id not in fused_scores:
                    fused_scores[chunk_id] = 0.0
                    chunk_map[chunk_id] = hit
                    
                fused_scores[chunk_id] += 1.0 / (k + rank + 1) # rank is 0-indexed
                
        process_list(dense_results)
        process_list(sparse_results)
        
        # Rebuild list sorted by RRF score
        fused_results = []
        for chunk_id, score in fused_scores.items():
            chunk = chunk_map[chunk_id].copy()
            chunk["rrf_score"] = score
            fused_results.append(chunk)
            
        fused_results.sort(key=lambda x: x["rrf_score"], reverse=True)
        return fused_results

    @staticmethod
    async def search(repository_id: str, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Executes a full hybrid search pipeline: Dense + Sparse -> RRF -> Cross-Encoder Reranking
        """
        try:
            # 1. Concurrent Dense and Sparse retrieval
            # We fetch top 50 from both sides to ensure good recall for RRF
            dense_task = dense_retriever.search(repository_id=repository_id, query=query, top_k=50)
            sparse_task = sparse_retriever.search(repository_id=repository_id, query=query, top_k=50)
            
            dense_results, sparse_results = await asyncio.gather(dense_task, sparse_task)
            
            # 2. Reciprocal Rank Fusion
            fused_results = HybridRetriever.rrf_fusion(dense_results, sparse_results)
            
            # 3. Cross Encoder Reranking
            # We rerank the top 50 fused results to save compute time
            candidates_to_rerank = fused_results[:50]
            final_results = cross_encoder_reranker.rerank(query=query, results=candidates_to_rerank, top_k=top_k)
            
            return final_results
            
        except Exception as e:
            logger.error(f"Hybrid retrieval failed for repo {repository_id}: {e}", exc_info=True)
            raise

hybrid_retriever = HybridRetriever()
