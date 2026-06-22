import logging
from fastapi import APIRouter, HTTPException, status
from app.schemas.api import ChatRequest, ChatResponse, Citation, BenchmarkRequest, BenchmarkResponse
from app.graph.workflow import rag_app
from app.retrieval.evaluator import retrieval_evaluator

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["Chat"])

@router.post("", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Core RAG endpoint.
    Retrieves context from Qdrant and generates an answer using the LLM provider.
    """
    if not request.repository_id or "/" not in request.repository_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid repository_id format. Must be owner/repo"
        )
        
    try:
        # 1. Initialize Graph State
        initial_state = {
            "repository_id": request.repository_id,
            "question": request.question,
            "search_queries": [],
            "retrieved_chunks": [],
            "context_chunks": [],
            "answer": "",
            "citations": []
        }
        
        # 2. Execute LangGraph Workflow
        final_state = await rag_app.ainvoke(initial_state)
        
        # 3. Format Response
        citations = [
            Citation(
                source_type=cit["source_type"],
                title=cit["title"],
                url=cit["url"]
            ) for cit in final_state.get("citations", [])
        ]
        
        return ChatResponse(
            answer=final_state.get("answer", "No answer generated."),
            citations=citations,
            repository_id=request.repository_id
        )
        
    except Exception as e:
        logger.error(f"Chat failed for {request.repository_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate answer."
        )

@router.post("/benchmark", response_model=BenchmarkResponse)
async def benchmark_endpoint(request: BenchmarkRequest):
    """
    Evaluates the retrieval quality of the Dense, Sparse, and Hybrid retrievers
    across a set of test queries.
    """
    try:
        from app.retrieval.evaluator import TestQuery
        test_queries = [
            TestQuery(
                query=q.query,
                expected_keywords=q.expected_keywords,
                expected_source_types=q.expected_source_types
            ) for q in request.test_queries
        ]
        
        result = await retrieval_evaluator.run_benchmark(request.repository_id, test_queries)
        return BenchmarkResponse(**result)
        
    except Exception as e:
        logger.error(f"Benchmark failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to run retrieval benchmark."
        )
