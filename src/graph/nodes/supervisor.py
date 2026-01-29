import sys
from enum import Enum
from typing import Any

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import create_model

from src.core.llm_factory import build_chat_llm
from src.graph.state import AgentState

# Define the set of workers
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


def _route_schema(options: list[str]):
    """
    Build a constrained Pydantic schema for structured output.

    We use Enum because Literal cannot be parameterized dynamically in python 3.10 syntax.
    """

    def _name(opt: str) -> str:
        return (opt or "UNKNOWN").upper().replace("-", "_")

    RouteEnum = Enum("RouteEnum", {_name(opt): opt for opt in options})
    return create_model("RouteResponse", next=(RouteEnum, ...))


class SupervisorNode:
    def __init__(self):
        # Deterministic routing: keep temperature at 0.
        self.llm = build_chat_llm(temperature=0.0)

    def __call__(self, state: AgentState):
        allowed = _normalized_allowed_workers(state)
        options = ["FINISH"] + allowed

        capabilities = "\n".join([f"- {w}: {_WORKER_CAPABILITIES.get(w, '')}".rstrip() for w in allowed])
        system_prompt = (
            "You are a supervisor tasked with managing a conversation between the following workers: {members}. "
            "Given the user request and the conversation so far, respond with the worker to act next. "
            "Each worker will perform a task and respond with their results. When finished, respond with FINISH.\n\n"
            "Worker capabilities:\n"
            f"{capabilities}"
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder(variable_name="messages"),
                (
                    "system",
                    "Given the conversation above, who should act next? Or should we FINISH? Select one of: {options}",
                ),
            ]
        ).partial(options=str(options), members=", ".join(allowed))

        RouteResponse = _route_schema(options)
        chain = prompt | self.llm.with_structured_output(RouteResponse)

        # The prompt only needs `messages`; other state keys are ignored.
        result: Any = chain.invoke({"messages": state["messages"]})
        next_ = getattr(result, "next", None)
        if hasattr(next_, "value"):
            next_ = next_.value
        return {"next": next_ or "FINISH"}


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
