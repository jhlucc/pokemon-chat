import operator
from collections.abc import Sequence
from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from typing_extensions import NotRequired, Required


class AgentState(TypedDict, total=False):
    """
    The state of the agent graph.
    """

    # The history of messages in the conversation
    messages: Required[Annotated[Sequence[BaseMessage], operator.add]]

    # The next node to route to (single worker or "FINISH")
    next: Required[str]

    # Optional routing constraints (set by the frontend in Agent mode)
    # Example: ["rag_worker", "graph_worker", "web_worker", "stats_worker", "mcp_worker"]
    allowed_workers: NotRequired[list[str]]
    # Optional: knowledge base selection (Milvus collection id)
    db_id: NotRequired[str]

    # When True, the worker's response is forwarded directly to output
    # without returning to the supervisor for re-evaluation.
    # Set by supervisor when rule-based routing gives a confident match.
    forward_directly: NotRequired[bool]

    # --- Parallel execution support ---
    # List of workers to execute in parallel. When set, supervisor uses Send() API.
    parallel_workers: NotRequired[list[str]]
    # Count of parallel workers dispatched (for tracking completion)
    parallel_count: NotRequired[int]
    # Count of parallel workers that have completed
    parallel_done: NotRequired[int]
