from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from threading import Lock
from typing import Any, Dict

from src.core.settings import settings

_lock = Lock()

# Only store non-sensitive, UI-driven preferences here.
_ALLOWED_PATCH_KEYS = {
    "model_provider",
    "model_name",
    # Feature flags (backend behavior)
    "enable_knowledge_base",
    "enable_knowledge_graph",
    "enable_web_search",
    "enable_mcp",
    "enable_reranker",
    "enable_asr",
    "enable_ner_bert",
}


def _config_dir() -> Path:
    return Path(settings.paths.save_yaml_path) / "config"


def _config_file() -> Path:
    return _config_dir() / "ui_config.json"


def _read_json_file(path: Path) -> Dict[str, Any]:
    try:
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        # Corrupt or unreadable config should not break server startup.
        return {}


def _atomic_write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, path)
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass


def load_ui_overrides() -> Dict[str, Any]:
    """Load persisted UI overrides (non-sensitive)."""
    with _lock:
        return _read_json_file(_config_file())


def patch_ui_overrides(patch: Dict[str, Any]) -> Dict[str, Any]:
    """Patch persisted UI overrides and return the new stored overrides."""
    patch = patch or {}
    safe_patch = {k: v for k, v in patch.items() if k in _ALLOWED_PATCH_KEYS}

    with _lock:
        cur = _read_json_file(_config_file())
        cur.update(safe_patch)
        _atomic_write_json(_config_file(), cur)
        # Invalidate runtime caches so changes apply immediately.
        try:
            from src.core.feature_flags import clear_feature_cache

            clear_feature_cache()
        except Exception:
            pass
        try:
            from src.core.llm_factory import clear_ui_overrides_cache

            clear_ui_overrides_cache()
        except Exception:
            pass
        return cur


def build_ui_config() -> Dict[str, Any]:
    """
    Build the config object consumed by the frontend.
    IMPORTANT: do NOT include secrets (API keys).
    """
    overrides = load_ui_overrides()

    model_provider = overrides.get("model_provider") or "siliconflow"
    model_name = overrides.get("model_name") or settings.llm.model_name

    def _get_bool(name: str, default: bool) -> bool:
        v = overrides.get(name, None)
        if v is None:
            return bool(default)
        if isinstance(v, bool):
            return v
        if isinstance(v, (int, float)):
            return bool(v)
        if isinstance(v, str):
            s = v.strip().lower()
            if s in {"1", "true", "yes", "y", "on"}:
                return True
            if s in {"0", "false", "no", "n", "off"}:
                return False
        return bool(default)

    return {
        "backend": {
            "online": True,
            "ready": None,
            "last_error": None,
            "checks": None,
        },
        "model_provider": model_provider,
        "model_name": model_name,
        # Feature flags are backend capabilities (read-only from UI perspective).
        "enable_knowledge_base": _get_bool("enable_knowledge_base", settings.features.enable_knowledge_base),
        "enable_knowledge_graph": _get_bool("enable_knowledge_graph", settings.features.enable_knowledge_graph),
        "enable_web_search": _get_bool("enable_web_search", settings.features.enable_web_search),
        "enable_mcp": _get_bool("enable_mcp", settings.features.enable_mcp),
        "enable_reranker": _get_bool("enable_reranker", settings.features.enable_reranker),
        "enable_asr": _get_bool("enable_asr", settings.features.enable_asr),
        "enable_ner_bert": _get_bool("enable_ner_bert", settings.features.enable_ner_bert),
        # Display-only model info
        "embed_model": settings.embedding.model_name,
        "reranker": settings.reranker.model_name,
        # Compatibility with existing UI code
        "custom_models": [],
    }

