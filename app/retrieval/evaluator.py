import logging
from typing import List, Dict, Any
from pydantic import BaseModel

from app.retrieval.dense_retriever import dense_retriever
from app.retrieval.sparse_retriever import sparse_retriever
from app.retrieval.hybrid_search import hybrid_retriever

logger = logging.getLogger(__name__)

class TestQuery(BaseModel):
    query: str
    expected_keywords: List[str]
    expected_source_types: List[str]

class RetrievalEvaluator:
    @staticmethod
    def _is_relevant(hit: Dict[str, Any], query_spec: TestQuery) -> bool:
        text = hit.get("text", "").lower()
        source_type = hit.get("source_type", "").lower()
        
        # Check source type match
        if query_spec.expected_source_types:
            if source_type in [t.lower() for t in query_spec.expected_source_types]:
                return True
                
        # Check keyword match
        if query_spec.expected_keywords:
            for kw in query_spec.expected_keywords:
                if kw.lower() in text:
                    return True
                    
        return False

    @staticmethod
    def _evaluate_results(results: List[Dict[str, Any]], query_spec: TestQuery, k: int = 5) -> float:
        """
        Returns Precision@K
        """
        top_k = results[:k]
        relevant_count = sum(1 for hit in top_k if RetrievalEvaluator._is_relevant(hit, query_spec))
        return relevant_count / k if k > 0 else 0.0

    @staticmethod
    async def evaluate_query(repository_id: str, query_spec: TestQuery, k: int = 5) -> Dict[str, Any]:
        """
        Evaluates a single query against dense, sparse, and hybrid retrievers.
        """
        try:
            query = query_spec.query
            
            # Run all retrievers concurrently
            dense_results = await dense_retriever.search(repository_id, query, top_k=k)
            sparse_results = await sparse_retriever.search(repository_id, query, top_k=k)
            hybrid_results = await hybrid_retriever.search(repository_id, query, top_k=k)
            
            dense_p_at_k = RetrievalEvaluator._evaluate_results(dense_results, query_spec, k)
            sparse_p_at_k = RetrievalEvaluator._evaluate_results(sparse_results, query_spec, k)
            hybrid_p_at_k = RetrievalEvaluator._evaluate_results(hybrid_results, query_spec, k)
            
            improvement = "neutral"
            if hybrid_p_at_k > dense_p_at_k and hybrid_p_at_k >= sparse_p_at_k:
                improvement = "positive"
            elif hybrid_p_at_k < dense_p_at_k or hybrid_p_at_k < sparse_p_at_k:
                improvement = "negative"
                
            return {
                "query": query,
                "dense_precision_at_k": dense_p_at_k,
                "bm25_precision_at_k": sparse_p_at_k,
                "hybrid_precision_at_k": hybrid_p_at_k,
                "improvement": improvement
            }
            
        except Exception as e:
            logger.error(f"Failed to evaluate query '{query_spec.query}': {e}", exc_info=True)
            return {
                "query": query_spec.query,
                "error": str(e)
            }

    @staticmethod
    async def run_benchmark(repository_id: str, test_queries: List[TestQuery]) -> Dict[str, Any]:
        """
        Runs batch evaluation across all test queries.
        """
        results = []
        for q in test_queries:
            res = await RetrievalEvaluator.evaluate_query(repository_id, q, k=5)
            results.append(res)
            
        # Aggregate metrics
        valid_results = [r for r in results if "error" not in r]
        
        avg_dense = sum(r["dense_precision_at_k"] for r in valid_results) / len(valid_results) if valid_results else 0.0
        avg_sparse = sum(r["bm25_precision_at_k"] for r in valid_results) / len(valid_results) if valid_results else 0.0
        avg_hybrid = sum(r["hybrid_precision_at_k"] for r in valid_results) / len(valid_results) if valid_results else 0.0
        
        best_strategy = "dense"
        if avg_sparse > avg_dense and avg_sparse > avg_hybrid:
            best_strategy = "bm25"
        elif avg_hybrid >= avg_dense and avg_hybrid >= avg_sparse:
            best_strategy = "hybrid"
            
        return {
            "results": results,
            "overall_metrics": {
                "avg_dense_precision_at_5": avg_dense,
                "avg_bm25_precision_at_5": avg_sparse,
                "avg_hybrid_precision_at_5": avg_hybrid,
                "best_strategy": best_strategy
            }
        }

retrieval_evaluator = RetrievalEvaluator()
