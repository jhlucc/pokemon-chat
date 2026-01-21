"""
src.knowledge.core

Keep this package import lightweight.

Historically this package performed eager star-imports of heavy modules (OCR, PDF
parsers, Milvus, torch, etc.) which caused slow and fragile server startup.

Prefer importing from submodules directly:
  - from src.knowledge.core.indexing import chunk_file, parse_file
  - from src.knowledge.core.history_chat import HistoryManager

For backward compatibility, a small set of commonly used symbols are exposed
via lazy attribute loading (PEP 562).
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    # History
    "HistoryManager",
    # Indexing
    "parse_file",
    "chunk_file",
    "chunk_text",
    # Milvus helper
    "MilvusStorage",
    # Vector recall
    "VectorRecaller",
    # Legacy helpers
    "get_kg_agent",
]


_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    # history_chat
    "HistoryManager": ("src.knowledge.core.history_chat", "HistoryManager"),
    # indexing
    "parse_file": ("src.knowledge.core.indexing", "parse_file"),
    "chunk_file": ("src.knowledge.core.indexing", "chunk_file"),
    "chunk_text": ("src.knowledge.core.indexing", "chunk_text"),
    # Milvus
    "MilvusStorage": ("src.knowledge.core.Milvus", "MilvusStorage"),
    # Vector recall
    "VectorRecaller": ("src.knowledge.core.vectorrecall", "VectorRecaller"),
}


def __getattr__(name: str) -> Any:  # pragma: no cover
    target = _LAZY_ATTRS.get(name)
    if not target:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    mod_name, attr_name = target
    mod = import_module(mod_name)
    value = getattr(mod, attr_name)
    globals()[name] = value  # cache on first use
    return value


def get_kg_agent():
    """Backward-compatible helper. Prefer `src.runtime.get_kg_agent()`."""
    from src.runtime import get_kg_agent as _get_kg_agent

    return _get_kg_agent()

