from __future__ import annotations

from dotenv import load_dotenv
from pathlib import Path

# Best-effort load of the project root `.env` (server/main.py also loads it).
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from concurrent.futures import ThreadPoolExecutor
executor = ThreadPoolExecutor()

# Lazy-load Config to keep `import src` cheap (tests/tools often import subpackages).
_config_instance = None


def get_config():
    """Get the global Config singleton (lazy)."""
    global _config_instance
    if _config_instance is None:
        from src.config import Config

        _config_instance = Config()
    return _config_instance


class _ConfigProxy:
    """Backward-compatible lazy proxy for the old `config` global."""

    def __getattr__(self, name):
        return getattr(get_config(), name)

    def __getitem__(self, key):
        return get_config().__getitem__(key)

    def __setitem__(self, key, value):
        return get_config().__setitem__(key, value)

    def __contains__(self, key):  # pragma: no cover
        return key in get_config()

    def __iter__(self):  # pragma: no cover
        return iter(get_config())

    def __len__(self):  # pragma: no cover
        return len(get_config())

    def __repr__(self) -> str:  # pragma: no cover
        return "<ConfigProxy lazy>"


# Backward-compatible alias (lazy).
config = _ConfigProxy()

# 延迟加载 KnowledgeBase，避免在导入时要求 API Key
_knowledge_base = None

def get_knowledge_base():
    """获取 KnowledgeBase 单例（延迟加载）"""
    global _knowledge_base
    if _knowledge_base is None:
        # Delegate to runtime singletons to avoid creating multiple KB instances.
        from src.runtime import get_kb
        _knowledge_base = get_kb()
    return _knowledge_base

# 保持向后兼容的别名
class _KnowledgeBaseProxy:
    """Lazy proxy for KnowledgeBase.

    Importing/initializing KnowledgeBase can be expensive (models, API keys, etc). This
    proxy keeps module import cheap while preserving the old `src.knowledge_base.xxx`
    call sites.
    """

    def __getattr__(self, name):
        return getattr(get_knowledge_base(), name)

    def __repr__(self) -> str:  # pragma: no cover
        return "<KnowledgeBaseProxy lazy>"

# Backward-compatible alias (lazy).
knowledge_base = _KnowledgeBaseProxy()

def get_retriever():
    # Preserve legacy API while using the runtime singleton.
    from src.runtime import get_retriever as _get_retriever
    return _get_retriever()
