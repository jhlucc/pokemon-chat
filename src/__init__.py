"""
Pokemon-Chat Package Root

This module provides lazy-loaded singletons and backward-compatible proxies.
All new code should import from specific submodules (e.g., `from src.core.settings import settings`).
"""

from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv

# Best-effort load of the project root `.env` (server/main.py also loads it).
# During pytest runs we avoid loading `.env` to keep tests deterministic/offline-safe.
if "pytest" not in sys.modules:
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from concurrent.futures import ThreadPoolExecutor

executor = ThreadPoolExecutor()


def shutdown_executor(*, wait: bool = False) -> None:
    """Best-effort shutdown for the shared threadpool."""
    global executor
    try:
        executor.shutdown(wait=wait, cancel_futures=True)
    except TypeError:
        # Older Python compatibility (no cancel_futures kwarg).
        executor.shutdown(wait=wait)


# =============================================================================
# DEPRECATED: Legacy config proxy (for backward compatibility only)
# New code should use `from src.core.settings import settings`
# =============================================================================
# Config proxy removed. Please use src.core.settings.settings.


# =============================================================================
# KnowledgeBase singleton
# =============================================================================
_knowledge_base = None


def get_knowledge_base():
    """获取 KnowledgeBase 单例（延迟加载）"""
    global _knowledge_base
    if _knowledge_base is None:
        from src.runtime import get_kb

        _knowledge_base = get_kb()
    return _knowledge_base


class _KnowledgeBaseProxy:
    """Lazy proxy for KnowledgeBase."""

    def __getattr__(self, name):
        return getattr(get_knowledge_base(), name)

    def __repr__(self) -> str:
        return "<KnowledgeBaseProxy lazy>"


knowledge_base = _KnowledgeBaseProxy()


def get_retriever():
    """Get the global Retriever singleton."""
    from src.runtime import get_retriever as _get_retriever

    return _get_retriever()
