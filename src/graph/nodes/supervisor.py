import sys
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool

from src.core.llm_factory import build_chat_llm
from src.graph.nodes.rule_router import rule_route
from src.graph.state import AgentState

# Definition of workers
AVAILABLE_WORKERS = ["rag_worker", "web_worker", "graph_worker", "stats_worker", "mcp_worker"]

_WORKER_CAPABILITIES: dict[str, str] = {
    "rag_worker": "Pokemon knowledge retrieval from vector database",
    "web_worker": "Real-time web search for current information",
    "graph_worker": "Knowledge graph queries (relationships, evolutions)",
    "stats_worker": "Pokemon stats and battle calculations",
    "mcp_worker": "Geographic location queries (real-world Pokemon locations)",
}


def _normalized_allowed_workers(state: AgentState) -> list[str]:
    allowed_raw = state.get("allowed_workers")
    if not allowed_raw:
        return list(AVAILABLE_WORKERS)
    if not isinstance(allowed_raw, list):
        return list(AVAILABLE_WORKERS)
    allowed = [w for w in allowed_raw if isinstance(w, str) and w in AVAILABLE_WORKERS]
    return allowed or list(AVAILABLE_WORKERS)


# ---- Handoff tools ----
# Each tool represents a routing decision. The supervisor calls one of these
# tools to indicate which worker should handle the request next.
# This leverages the LLM's native tool-calling for more reliable routing.


def _build_handoff_tools(allowed: list[str]) -> list:
    """Build handoff tools dynamically based on allowed workers."""
    tools = []
    for worker in allowed:
        cap = _WORKER_CAPABILITIES.get(worker, "")

        @tool(name=f"route_to_{worker}", description=f"Hand off to {worker}: {cap}")
        def _handoff(reason: str = "") -> str:  # noqa: ARG001
            """Route to this worker. Provide a brief reason for the routing decision."""
            return "routed"

        tools.append(_handoff)

    # FINISH tool signals the conversation is complete
    @tool(name="finish", description="End the conversation. Use when the task is complete or already answered.")
    def _finish(reason: str = "") -> str:  # noqa: ARG001
        """Finish the conversation."""
        return "finished"

    tools.append(_finish)
    return tools


def _parse_tool_call_route(response) -> str:
    """Extract the routing decision from the LLM's tool call response."""
    tool_calls = getattr(response, "tool_calls", None)
    if not tool_calls:
        return "FINISH"

    tool_name = tool_calls[0]["name"]

    if tool_name == "finish":
        return "FINISH"

    # Extract worker name from "route_to_{worker}" pattern
    if tool_name.startswith("route_to_"):
        worker = tool_name[len("route_to_"):]
        if worker in AVAILABLE_WORKERS:
            return worker

    return "FINISH"


class SupervisorNode:
    def __init__(self):
        # Deterministic routing: keep temperature at 0.
        self.llm = build_chat_llm(temperature=0.0)

    def __call__(self, state: AgentState):
        allowed = _normalized_allowed_workers(state)

        # Fast-path: when a simple heuristic can confidently route the request,
        # skip LLM routing entirely to reduce misroutes and improve latency.
        last_human: str | None = None
        for msg in reversed(state.get("messages") or []):
            if isinstance(msg, HumanMessage):
                last_human = msg.content
                break
            if isinstance(msg, BaseMessage) and getattr(msg, "type", None) == "human":
                last_human = getattr(msg, "content", None)
                break

        # --- Forward-directly fast-path ---
        # When we return here after a worker has responded, check if
        # forward_directly was set on the initial routing pass.
        # If so, the worker's response is already in messages – finish immediately
        # without an extra LLM call.
        if state.get("forward_directly") and len(state.get("messages", [])) > 1:
            return {"next": "FINISH"}

        route = rule_route(last_human or "", allowed)
        if route:
            # Rule-based match is high-confidence → mark for direct forwarding
            return {"next": route, "forward_directly": True}

        # --- Tool-based LLM routing ---
        # Bind handoff tools to the LLM and let it choose via native tool calling.
        # This is more reliable than structured output for routing decisions.
        handoff_tools = _build_handoff_tools(allowed)
        llm_with_tools = self.llm.bind_tools(handoff_tools, tool_choice="required")

        capabilities = "\n".join([f"- {w}: {_WORKER_CAPABILITIES.get(w, '')}".rstrip() for w in allowed])
        system_prompt = (
            "You are a supervisor managing specialized workers. "
            "Given the user request and conversation, call ONE tool to route to the appropriate worker. "
            "Each worker will perform a task and respond with results. "
            "When the task is already answered or finished, call the finish tool.\n\n"
            "Worker capabilities:\n"
            f"{capabilities}"
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        chain = prompt | llm_with_tools

        result: Any = chain.invoke({"messages": state["messages"]})
        next_ = _parse_tool_call_route(result)
        return {"next": next_}


_supervisor_node: SupervisorNode | None = None


def get_supervisor_node() -> SupervisorNode:
    """
    Cached node instance.

    NOTE: tests patch classes heavily; avoid caching under pytest to keep patches effective.
    """
    if "pytest" in sys.modules:
        return SupervisorNode()
    global _supervisor_node
    if _supervisor_node is None:
        _supervisor_node = SupervisorNode()
    return _supervisor_node


def clear_supervisor_node_cache() -> None:
    global _supervisor_node
    _supervisor_node = None


def supervisor_node(state: AgentState):
    node = get_supervisor_node()
    return node(state)
