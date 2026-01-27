from langgraph.graph import END, StateGraph

from src.graph.nodes.graph_worker import graph_worker_node
from src.graph.nodes.mcp_worker import get_mcp_worker
from src.graph.nodes.rag_worker import rag_worker_node
from src.graph.nodes.stats_worker import stats_worker_node
from src.graph.nodes.supervisor import supervisor_node
from src.graph.nodes.web_worker import web_worker_node
from src.graph.state import AgentState

# Initialize Graph
workflow = StateGraph(AgentState)


# MCP Worker (async, needs wrapper)
async def mcp_worker_node(state: AgentState):
    worker = get_mcp_worker()
    return await worker(state)


# Add Nodes
workflow.add_node("supervisor", supervisor_node)
workflow.add_node("rag_worker", rag_worker_node)
workflow.add_node("web_worker", web_worker_node)
workflow.add_node("graph_worker", graph_worker_node)
workflow.add_node("stats_worker", stats_worker_node)
workflow.add_node("mcp_worker", mcp_worker_node)

# Add Edges: Workers return to Supervisor
workflow.add_edge("rag_worker", "supervisor")
workflow.add_edge("web_worker", "supervisor")
workflow.add_edge("graph_worker", "supervisor")
workflow.add_edge("stats_worker", "supervisor")
workflow.add_edge("mcp_worker", "supervisor")

# Conditional Edge for Supervisor
workflow.add_conditional_edges(
    "supervisor",
    lambda x: x["next"],
    {
        "rag_worker": "rag_worker",
        "web_worker": "web_worker",
        "graph_worker": "graph_worker",
        "stats_worker": "stats_worker",
        "mcp_worker": "mcp_worker",
        "FINISH": END,
    },
)

# Entry Point
workflow.set_entry_point("supervisor")

# Compile
# checkpointer = MemorySaver() # TODO: Add persistence if needed
graph = workflow.compile()
