from __future__ import annotations

"""
Centralized helpers for constructing OpenAI-compatible LangChain clients.

Why:
- Many modules instantiated ChatOpenAI directly using `settings.llm.*`.
- The project also supports provider-specific credentials via the Settings UI
  (stored under resources/save/config) and env compatibility keys.
- A single factory keeps credential resolution consistent and avoids import-time
  surprises across graph workers, retrievers, etc.
"""

import sys
import json
from functools import lru_cache
from pathlib import Path
from typing import Optional

from langchain_openai import ChatOpenAI

from src.core.provider_config import get_provider_api_base, get_provider_api_key
from src.core.settings import settings
from src.utils.http_client import get_safe_httpx_client
from src.utils.logger import get_logger

_log = get_logger(__name__)


def _ui_config_file() -> Path:
    return Path(settings.paths.save_yaml_path) / "config" / "ui_config.json"


@lru_cache(maxsize=1)
def _load_ui_overrides_cached() -> dict:
    # Keep tests deterministic/offline-safe: never read local UI state from disk.
    if "pytest" in sys.modules:
        return {}
    path = _ui_config_file()
    try:
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def clear_ui_overrides_cache() -> None:
    _load_ui_overrides_cached.cache_clear()


def build_chat_llm(
    *,
    model_provider: Optional[str] = None,
    model_name: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> ChatOpenAI:
    """
    Build a ChatOpenAI instance using the project's provider resolution rules.

    Notes:
    - We intentionally do NOT hard-fail when api_key is empty, because:
      - The server/UI can run in offline/mock mode.
      - Unit tests patch the classes that call into the network.
    - Call sites that *require* online LLM access should validate readiness (see /readyz).
    """

    overrides = _load_ui_overrides_cached()
    provider = (model_provider or overrides.get("model_provider") or "siliconflow").strip().lower()
    name = (model_name or overrides.get("model_name") or settings.llm.model_name).strip() or settings.llm.model_name

    api_key = (
        get_provider_api_key(provider)
        or settings.get_api_key(provider)
        or settings.llm.api_key
        or ""
    ).strip()
    base_url = (
        get_provider_api_base(provider)
        or settings.llm.api_base
        or ""
    ).strip()

    if not api_key and "pytest" not in sys.modules:
        _log.warning(f"LLM api_key is empty for provider='{provider}'. Calls may fail until configured.")

    return ChatOpenAI(
        model=name,
        api_key=api_key,
        base_url=base_url,
        temperature=settings.llm.temperature if temperature is None else temperature,
        max_tokens=settings.llm.max_tokens if max_tokens is None else max_tokens,
        openai_proxy=None,
        http_client=get_safe_httpx_client(),
    )
