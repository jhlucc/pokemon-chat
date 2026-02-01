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


def _post_worker_route(state: AgentState) -> str:
    """Route after a worker finishes.

    If the supervisor set ``forward_directly`` during rule-based routing,
    skip the return trip to the supervisor and go straight to END.
    This saves one LLM routing call and avoids potential paraphrasing.
    """
    if state.get("forward_directly"):
        return "end"
    return "supervisor"


# Add Nodes
workflow.add_node("supervisor", supervisor_node)
workflow.add_node("rag_worker", rag_worker_node)
workflow.add_node("web_worker", web_worker_node)
workflow.add_node("graph_worker", graph_worker_node)
workflow.add_node("stats_worker", stats_worker_node)
workflow.add_node("mcp_worker", mcp_worker_node)

# Add Edges: Workers route through _post_worker_route
# If forward_directly is set, go to END; otherwise return to supervisor.
_worker_route_map = {"supervisor": "supervisor", "end": END}

for worker_name in ["rag_worker", "web_worker", "graph_worker", "stats_worker", "mcp_worker"]:
    workflow.add_conditional_edges(worker_name, _post_worker_route, _worker_route_map)

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
graph = workflow.compile()
