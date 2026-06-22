from langgraph.graph import StateGraph, END

from app.graph.state import GraphState
from app.graph.nodes import (
    planner_node,
    expansion_node,
    retrieval_node,
    context_builder_node,
    generation_node,
    tool_agent_node,
    citation_node
)
import os

CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.0"))

# 1. Initialize the StateGraph
workflow = StateGraph(GraphState)

# 2. Add nodes
workflow.add_node("planner", planner_node)
workflow.add_node("expansion", expansion_node)
workflow.add_node("retrieval", retrieval_node)
workflow.add_node("context_builder", context_builder_node)
workflow.add_node("generation", generation_node)
workflow.add_node("tool_agent", tool_agent_node)
workflow.add_node("citation", citation_node)

def route_after_retrieval(state: GraphState) -> str:
    if state.get("retrieval_confidence", 0.0) >= CONFIDENCE_THRESHOLD:
        return "context_builder"
    return "tool_agent"

# 3. Define the edges
workflow.set_entry_point("planner")
workflow.add_edge("planner", "expansion")
workflow.add_edge("expansion", "retrieval")
workflow.add_conditional_edges(
    "retrieval",
    route_after_retrieval,
    {
        "context_builder": "context_builder",
        "tool_agent": "tool_agent"
    }
)
workflow.add_edge("context_builder", "generation")
workflow.add_edge("generation", "citation")
workflow.add_edge("tool_agent", "citation")
workflow.add_edge("citation", END)

# 4. Compile the graph
rag_app = workflow.compile()
