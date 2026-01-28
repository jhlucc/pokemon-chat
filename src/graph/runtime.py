from __future__ import annotations

from src.utils.logger import get_logger

_log = get_logger(__name__)


def reset_graph_workers() -> None:
    """
    Clear cached LangGraph worker singletons.

    This is useful when:
    - the UI changes model provider/name and we want new workers to pick it up
    - credentials/base URLs change via /providers
    """
    cleared: list[str] = []

    try:
        from src.graph.nodes.supervisor import clear_supervisor_node_cache

        clear_supervisor_node_cache()
        cleared.append("supervisor")
    except Exception:  # noqa: BLE001
        pass

    try:
        from src.graph.nodes.rag_worker import clear_rag_worker_cache

        clear_rag_worker_cache()
        cleared.append("rag_worker")
    except Exception:  # noqa: BLE001
        pass

    try:
        from src.graph.nodes.web_worker import clear_web_worker_cache

        clear_web_worker_cache()
        cleared.append("web_worker")
    except Exception:  # noqa: BLE001
        pass

    try:
        from src.graph.nodes.graph_worker import clear_graph_worker_cache

        clear_graph_worker_cache()
        cleared.append("graph_worker")
    except Exception:  # noqa: BLE001
        pass

    try:
        from src.graph.nodes.stats_worker import clear_stats_worker_cache

        clear_stats_worker_cache()
        cleared.append("stats_worker")
    except Exception:  # noqa: BLE001
        pass

    # Optional: MCP worker is already a singleton; clear it if available.
    try:
        from src.graph.nodes.mcp_worker import clear_mcp_worker_cache

        clear_mcp_worker_cache()
        cleared.append("mcp_worker")
    except Exception:  # noqa: BLE001
        pass

    if cleared:
        _log.info(f"Graph worker caches cleared: {', '.join(cleared)}")

