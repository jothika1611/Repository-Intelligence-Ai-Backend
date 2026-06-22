import logging
from typing import Dict, Any

from app.graph.state import GraphState
from app.retrieval.hybrid_search import hybrid_retriever
from app.llm.provider import get_llm_provider
from app.llm.prompts import PromptBuilder
from app.llm.tools import get_github_tools
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
import os

logger = logging.getLogger(__name__)

async def planner_node(state: GraphState) -> Dict[str, Any]:
    """
    Analyzes the user question. For the MVP, this simply passes the original
    question directly to the expansion phase, but serves as an extension point
    for routing (e.g., code search vs doc search).
    """
    logger.info(f"Graph execution started for question: {state['question']}")
    return {"search_queries": [state["question"]]}

async def expansion_node(state: GraphState) -> Dict[str, Any]:
    """
    Generates semantic search variants.
    For the MVP, we just ensure the original query is used.
    Future iterations could use an LLM here to generate query permutations.
    """
    return {"search_queries": state.get("search_queries", [state["question"]])}

async def retrieval_node(state: GraphState) -> Dict[str, Any]:
    """
    Executes Hybrid Retrieval (Dense + Sparse + RRF + Reranker) for the queries.
    """
    all_results = []
    seen_ids = set()
    
    # We execute hybrid search for the first query in MVP
    query = state["search_queries"][0]
    
    try:
        results = await hybrid_retriever.search(
            repository_id=state["repository_id"],
            query=query,
            top_k=5
        )
        
        # Deduplicate
        for hit in results:
            if hit["id"] not in seen_ids:
                all_results.append(hit)
                seen_ids.add(hit["id"])
                
    except Exception as e:
        logger.error(f"Retrieval node failed: {e}", exc_info=True)
        
    # Calculate retrieval confidence (average cross_encoder_score)
    confidence = 0.0
    if all_results:
        scores = [hit.get("cross_encoder_score", 0.0) for hit in all_results]
        confidence = sum(scores) / len(scores)
        
    return {"retrieved_chunks": all_results, "retrieval_confidence": confidence}

async def context_builder_node(state: GraphState) -> Dict[str, Any]:
    """
    Formats the top-ranked retrieved chunks into a strict string structure.
    """
    context_chunks = []
    
    for idx, hit in enumerate(state.get("retrieved_chunks", [])):
        source_type = hit.get("source_type", "UNKNOWN").upper()
        title = hit.get("title", f"Document {idx}")
        text = hit.get("text", "")
        
        chunk_str = f"[Source Type: {source_type}]\nTitle: {title}\nContent:\n{text}"
        context_chunks.append(chunk_str)
        
    return {"context_chunks": context_chunks}

async def generation_node(state: GraphState) -> Dict[str, Any]:
    """
    Calls the LLM Provider using the PromptBuilder to generate the final answer.
    """
    if not state.get("context_chunks"):
        return {"answer": "I couldn't find any information about this in the repository."}
        
    llm_provider = get_llm_provider()
    
    try:
        answer = await llm_provider.generate(
            prompt=state["question"],
            context=state["context_chunks"]
        )
        return {"answer": answer}
    except Exception as e:
        logger.error(f"Generation node failed: {e}", exc_info=True)
        return {"answer": "An error occurred while generating the answer."}

async def tool_agent_node(state: GraphState) -> Dict[str, Any]:
    """
    Fallback agent node that uses LangGraph's create_react_agent to actively browse the repository.
    """
    logger.info("Retrieval confidence low. Routing to tool agent...")
    llm_provider = get_llm_provider()
    raw_llm = llm_provider.llm  # Assuming the provider exposes the underlying BaseChatModel as .llm
    
    tools = get_github_tools(state["repository_id"])
    
    # We create an ephemeral react agent just for this query
    agent = create_react_agent(raw_llm, tools)
    
    system_prompt = (
        f"You are a helpful assistant analyzing the repository {state['repository_id']}. "
        "Please use your tools to find the answer to the user's question. "
        "IMPORTANT: If you try to read a file and it is not found, the user might have made a spelling mistake (e.g., 'read me' instead of 'README.md'). "
        "In such cases, use the get_repository_tree() tool to look for similar filenames before giving up. "
        "If you still cannot find it, ask the user to clarify which file they meant."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": state["question"]}
    ]
    
    try:
        # Prevent infinite loops by setting a recursion limit
        config = {"recursion_limit": 10}
        result = await agent.ainvoke({"messages": messages}, config)
        answer = result["messages"][-1].content
        return {"answer": answer}
    except Exception as e:
        logger.error(f"Tool agent failed: {e}", exc_info=True)
        return {"answer": "I encountered an internal error while searching the repository. If you are asking about a specific file (like the README), please make sure the spelling is correct and try again!"}

async def citation_node(state: GraphState) -> Dict[str, Any]:
    """
    Extracts deterministic citations based only on retrieved chunks.
    """
    citations = []
    seen_urls = set()
    
    # If the answer explicitly states no info found, clear citations
    if "couldn't find any information" in state.get("answer", ""):
        return {"citations": []}
    
    for hit in state.get("retrieved_chunks", []):
        url = hit.get("url", "")
        if url and url not in seen_urls:
            citations.append({
                "source_type": hit.get("source_type", "UNKNOWN").upper(),
                "title": hit.get("title", "Unknown Title"),
                "url": url
            })
            seen_urls.add(url)
            
    return {"citations": citations}
