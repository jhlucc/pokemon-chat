"""
Centralized lazy singletons for runtime services.

Why:
- Many modules created heavyweight clients at import time (Neo4j/Milvus/LLM/etc),
  which makes `import server.main` slow and fragile when optional services are
  not running.
- FastAPI routers should request these services at runtime (per request/startup),
  not during module import.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

from src.utils.logger import get_logger

_log = get_logger(__name__)

_kb_instance = None
_retriever_instance = None
_kg_agent_instance = None
_graph_db_instance = None
_whisper_client_instance = None
_mcp_client_instance = None


# ---------------------------------------------------------------------------
# Lazy singletons
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def get_kb():
    """KnowledgeBase singleton (Milvus-backed)."""
    global _kb_instance
    from src.knowledge.store import KnowledgeBase

    _kb_instance = KnowledgeBase()
    return _kb_instance


@lru_cache(maxsize=1)
def get_retriever():
    """Retriever singleton (KG/KBase/Web/MCP orchestration)."""
    global _retriever_instance
    from src.knowledge.core.retriever import Retriever

    _retriever_instance = Retriever()
    return _retriever_instance


@lru_cache(maxsize=1)
def get_kg_agent():
    """Neo4j KGQueryAgent singleton."""
    global _kg_agent_instance
    from src.agents.kg_agent import KGQueryAgent

    _kg_agent_instance = KGQueryAgent()
    return _kg_agent_instance


@lru_cache(maxsize=1)
def get_graph_db():
    """Neo4j GraphDatabase singleton (graph management/embedding helpers)."""
    global _graph_db_instance
    from src.knowledge.store.graphbase import GraphDatabase

    _graph_db_instance = GraphDatabase()
    return _graph_db_instance


@lru_cache(maxsize=1)
def get_whisper_client():
    """Whisper ASR HTTP client singleton."""
    global _whisper_client_instance
    from src.utils.whisper_asr import WhisperClient

    _whisper_client_instance = WhisperClient()
    return _whisper_client_instance


@lru_cache(maxsize=1)
def get_mcp_client():
    """MCP client singleton."""
    global _mcp_client_instance
    from src.mcp.client_core import MCPClient

    _mcp_client_instance = MCPClient()
    return _mcp_client_instance


# ---------------------------------------------------------------------------
# Cache reset helpers (useful for /restart and tests)
# ---------------------------------------------------------------------------


def reset_all():
    """Clear all cached singletons."""
    global _kb_instance, _retriever_instance, _kg_agent_instance, _graph_db_instance, _whisper_client_instance, _mcp_client_instance
    get_kb.cache_clear()
    get_retriever.cache_clear()
    get_kg_agent.cache_clear()
    get_graph_db.cache_clear()
    get_whisper_client.cache_clear()
    get_mcp_client.cache_clear()
    _kb_instance = None
    _retriever_instance = None
    _kg_agent_instance = None
    _graph_db_instance = None
    _whisper_client_instance = None
    _mcp_client_instance = None
    _log.info("Runtime singletons cleared.")


async def aclose_all() -> None:
    """Best-effort close for resources that have a close/async-close API."""
    global _graph_db_instance, _mcp_client_instance

    if _graph_db_instance is not None and hasattr(_graph_db_instance, "close"):
        try:
            _graph_db_instance.close()
        except Exception as e:
            _log.warning(f"Graph DB close failed (ignored): {e}")

    if _mcp_client_instance is not None and hasattr(_mcp_client_instance, "aclose"):
        try:
            await _mcp_client_instance.aclose()
        except Exception as e:
            _log.warning(f"MCP client close failed (ignored): {e}")
