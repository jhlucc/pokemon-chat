from __future__ import annotations

import json
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import Any, Dict

from src.core.settings import settings

_lock = Lock()


def _config_dir() -> Path:
    return Path(settings.paths.save_yaml_path) / "config"


def _ui_config_file() -> Path:
    # Shared with `server/runtime_config.py` (UI overrides persisted by /config PATCH).
    return _config_dir() / "ui_config.json"


def _read_json_file(path: Path) -> Dict[str, Any]:
    try:
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        # Corrupt/unreadable config should never break runtime.
        return {}


@lru_cache(maxsize=1)
def _load_overrides_cached() -> Dict[str, Any]:
    return _read_json_file(_ui_config_file())


def clear_feature_cache() -> None:
    _load_overrides_cached.cache_clear()


def load_overrides() -> Dict[str, Any]:
    """
    Load persisted UI overrides.

    IMPORTANT:
    - This file is written by `/config` PATCH.
    - It may include feature flags that should affect backend behavior.
    """
    with _lock:
        return deepcopy(_load_overrides_cached())


def _as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"1", "true", "yes", "y", "on"}:
            return True
        if v in {"0", "false", "no", "n", "off"}:
            return False
    return default


def feature_enabled(key: str) -> bool:
    """
    Get the effective feature flag value.

    Precedence:
      1) persisted UI override (resources/save/config/ui_config.json)
      2) env/.env default via `settings.features.*`
    """
    key = (key or "").strip()
    if not key.startswith("enable_"):
        raise ValueError("feature key must start with 'enable_'")

    overrides = load_overrides()
    if key in overrides:
        return _as_bool(overrides.get(key), default=False)

    # Fallback to settings defaults.
    default = bool(getattr(settings.features, key, False))
    return default

