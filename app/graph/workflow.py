from langgraph.graph import StateGraph, END

from app.graph.state import GraphState
from app.graph.nodes import (
    planner_node,
    expansion_node,
    retrieval_node,
    context_builder_node,
    generation_node,
    citation_node
)

# 1. Initialize the StateGraph
workflow = StateGraph(GraphState)

# 2. Add nodes
workflow.add_node("planner", planner_node)
workflow.add_node("expansion", expansion_node)
workflow.add_node("retrieval", retrieval_node)
workflow.add_node("context_builder", context_builder_node)
workflow.add_node("generation", generation_node)
workflow.add_node("citation", citation_node)

# 3. Define the edges (Sequential MVP Execution)
workflow.set_entry_point("planner")
workflow.add_edge("planner", "expansion")
workflow.add_edge("expansion", "retrieval")
workflow.add_edge("retrieval", "context_builder")
workflow.add_edge("context_builder", "generation")
workflow.add_edge("generation", "citation")
workflow.add_edge("citation", END)

# 4. Compile the graph
rag_app = workflow.compile()
