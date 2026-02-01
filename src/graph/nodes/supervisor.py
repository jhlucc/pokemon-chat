import sys
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool

from src.core.llm_factory import build_chat_llm
from src.graph.nodes.rule_router import rule_route, rule_route_parallel
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


def _get_last_human_message(state: AgentState) -> str | None:
    """Extract the last human message from state."""
    for msg in reversed(state.get("messages") or []):
        if isinstance(msg, HumanMessage):
            return msg.content
        if isinstance(msg, BaseMessage) and getattr(msg, "type", None) == "human":
            return getattr(msg, "content", None)
    return None


# ---- Handoff tools ----
# Each tool represents a routing decision. The supervisor calls one of these
# tools to indicate which worker should handle the request next.
# This leverages the LLM's native tool-calling for more reliable routing.


def _build_handoff_tools(allowed: list[str], support_parallel: bool = True) -> list:
    """Build handoff tools dynamically based on allowed workers."""
    tools = []

    def _make_handoff_tool(worker_name: str, capability: str):
        """Create a handoff tool with proper closure binding."""

        @tool(name=f"route_to_{worker_name}", description=f"Hand off to {worker_name}: {capability}")
        def _handoff(reason: str = "") -> str:  # noqa: ARG001
            """Route to this worker. Provide a brief reason for the routing decision."""
            return f"routed to {worker_name}"

        return _handoff

    for worker in allowed:
        cap = _WORKER_CAPABILITIES.get(worker, "")
        tools.append(_make_handoff_tool(worker, cap))

    # FINISH tool signals the conversation is complete
    @tool(name="finish", description="End the conversation. Use when the task is complete or already answered.")
    def _finish(reason: str = "") -> str:  # noqa: ARG001
        """Finish the conversation."""
        return "finished"

    tools.append(_finish)

    # Parallel routing tool - calls multiple workers at once
    if support_parallel and len(allowed) > 1:

        @tool(
            name="route_parallel",
            description=(
                "Route to MULTIPLE workers in parallel. Use when the query needs data from "
                "multiple sources (e.g., stats AND evolution info). Provide comma-separated worker names."
            ),
        )
        def _parallel(workers: str, reason: str = "") -> str:  # noqa: ARG001
            """Route to multiple workers simultaneously. workers: comma-separated list like 'stats_worker,graph_worker'"""
            return "parallel"

        tools.append(_parallel)

    return tools


def _parse_tool_call_route(response, allowed: list[str]) -> dict[str, Any]:
    """Extract the routing decision from the LLM's tool call response.

    Returns a dict with either:
    - {"next": "worker_name"} for single worker
    - {"next": "__PARALLEL__", "parallel_workers": [...]} for parallel execution
    - {"next": "FINISH"} to end
    """
    tool_calls = getattr(response, "tool_calls", None)
    if not tool_calls or len(tool_calls) == 0:
        return {"next": "FINISH"}

    tool_name = tool_calls[0].get("name", "")
    tool_args = tool_calls[0].get("args", {})

    if tool_name == "finish":
        return {"next": "FINISH"}

    # Parallel routing
    if tool_name == "route_parallel":
        workers_str = tool_args.get("workers", "")
        workers = [w.strip() for w in workers_str.split(",") if w.strip() in allowed]
        if len(workers) > 1:
            return {"next": "__PARALLEL__", "parallel_workers": workers}
        elif len(workers) == 1:
            return {"next": workers[0]}
        return {"next": "FINISH"}

    # Single worker routing: extract from "route_to_{worker}"
    if tool_name.startswith("route_to_"):
        worker = tool_name[len("route_to_") :]
        if worker in AVAILABLE_WORKERS:
            return {"next": worker}

    return {"next": "FINISH"}


class SupervisorNode:
    def __init__(self):
        # Deterministic routing: keep temperature at 0.
        self.llm = build_chat_llm(temperature=0.0)

    def __call__(self, state: AgentState) -> dict[str, Any]:
        allowed = _normalized_allowed_workers(state)
        last_human = _get_last_human_message(state)

        # --- Forward-directly fast-path ---
        # When we return here after a worker has responded, check if
        # forward_directly was set on the initial routing pass.
        # If so, the worker's response is already in messages – finish immediately
        # without an extra LLM call.
        if state.get("forward_directly") and len(state.get("messages", [])) > 1:
            return {"next": "FINISH"}

        # --- Parallel completion check ---
        # If parallel workers were dispatched, check completion
        parallel_count = state.get("parallel_count", 0)
        parallel_done = state.get("parallel_done", 0)
        if parallel_count > 0 and parallel_done >= parallel_count:
            # All parallel workers done, finish
            return {"next": "FINISH", "forward_directly": True}

        # --- Rule-based parallel routing (fast path) ---
        parallel_workers = rule_route_parallel(last_human or "", allowed)
        if len(parallel_workers) > 1:
            # Multiple workers detected via rules → parallel execution
            return {
                "next": "__PARALLEL__",
                "parallel_workers": parallel_workers,
                "parallel_count": len(parallel_workers),
                "parallel_done": 0,
                "forward_directly": True,  # Skip re-evaluation after parallel completion
            }

        # --- Rule-based single routing (fast path) ---
        route = rule_route(last_human or "", allowed)
        if route:
            return {"next": route, "forward_directly": True}

        # --- Tool-based LLM routing ---
        # Bind handoff tools to the LLM and let it choose via native tool calling.
        handoff_tools = _build_handoff_tools(allowed, support_parallel=True)
        llm_with_tools = self.llm.bind_tools(handoff_tools, tool_choice="required")

        capabilities = "\n".join([f"- {w}: {_WORKER_CAPABILITIES.get(w, '')}".rstrip() for w in allowed])
        system_prompt = (
            "You are a supervisor managing specialized workers. "
            "Given the user request, call the appropriate routing tool.\n\n"
            "IMPORTANT: If the query needs information from MULTIPLE workers "
            "(e.g., both stats AND evolution), use route_parallel with comma-separated worker names.\n\n"
            "Worker capabilities:\n"
            f"{capabilities}\n\n"
            "When the task is complete, call finish."
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        chain = prompt | llm_with_tools

        result = chain.invoke({"messages": state["messages"]})
        parsed = _parse_tool_call_route(result, allowed)

        # Add parallel tracking if needed
        if parsed.get("next") == "__PARALLEL__":
            workers = parsed.get("parallel_workers", [])
            parsed["parallel_count"] = len(workers)
            parsed["parallel_done"] = 0
            parsed["forward_directly"] = True

        return parsed


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


def supervisor_node(state: AgentState) -> dict[str, Any]:
    node = get_supervisor_node()
    return node(state)
