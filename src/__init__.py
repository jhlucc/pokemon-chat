"""
Pokemon-Chat Package Root

This module provides lazy-loaded singletons and backward-compatible proxies.
All new code should import from specific submodules (e.g., `from src.core.settings import settings`).
"""
from __future__ import annotations

from dotenv import load_dotenv
from pathlib import Path

# Best-effort load of the project root `.env` (server/main.py also loads it).
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from concurrent.futures import ThreadPoolExecutor
executor = ThreadPoolExecutor()


# =============================================================================
# DEPRECATED: Legacy config proxy (for backward compatibility only)
# New code should use `from src.core.settings import settings`
# =============================================================================
_config_instance = None


def get_config():
    """Get the global Config singleton (lazy). DEPRECATED - use settings instead."""
    global _config_instance
    if _config_instance is None:
        from src.config import Config
        _config_instance = Config()
    return _config_instance


class _ConfigProxy:
    """Backward-compatible lazy proxy for the old `config` global. DEPRECATED."""

    def __getattr__(self, name):
        return getattr(get_config(), name)

    def __getitem__(self, key):
        return get_config().__getitem__(key)

    def __setitem__(self, key, value):
        return get_config().__setitem__(key, value)

    def __contains__(self, key):
        return key in get_config()

    def __iter__(self):
        return iter(get_config())

    def __len__(self):
        return len(get_config())

    def __repr__(self) -> str:
        return "<ConfigProxy lazy - DEPRECATED>"


# Backward-compatible alias (lazy). DEPRECATED.
config = _ConfigProxy()


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

