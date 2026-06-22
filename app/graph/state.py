from typing import TypedDict, List, Dict, Any, Optional

class GraphState(TypedDict):
    """
    Represents the state of our RAG application through the LangGraph pipeline.
    """
    repository_id: str
    question: str
    
    # Generated intermediate state
    search_queries: List[str]
    retrieved_chunks: List[Dict[str, Any]]
    context_chunks: List[str]
    
    # Final output
    answer: str
    citations: List[Dict[str, str]]
