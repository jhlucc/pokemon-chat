from __future__ import annotations

import json
import os
import tempfile
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Optional

from src.core.settings import settings

_lock = Lock()


_ALIAS_TO_CANONICAL: dict[str, str] = {
    # Frontend uses `zhipu`, but assets/env commonly use `zhipuai`.
    "zhipuai": "zhipu",
    # Some UIs/assets use these keys; keep compatibility.
    "together.ai": "togetherai",
    "together": "togetherai",
    "openrouter": "openrouterai",
}


def _canonical_provider(provider: str) -> str:
    p = (provider or "").strip().lower()
    return _ALIAS_TO_CANONICAL.get(p, p)


# Provider -> env var mapping (compat with existing `.env.template` and code)
_PROVIDER_ENV: dict[str, dict[str, str]] = {
    "siliconflow": {"api_key": "SILICONFLOW_API_KEY", "api_base": "SILICONFLOW_API_BASE"},
    "openai": {"api_key": "OPENAI_API_KEY", "api_base": "OPENAI_API_BASE"},
    "deepseek": {"api_key": "DEEPSEEK_API_KEY", "api_base": "DEEPSEEK_BASE_URL"},
    "zhipu": {"api_key": "ZHIPUAI_API_KEY", "api_base": "ZHIPUAI_API_BASE"},
    "openrouterai": {"api_key": "OPENROUTER_API_KEY", "api_base": "OPENROUTER_BASE_URL"},
    "dashscope": {"api_key": "DASHSCOPE_API_KEY", "api_base": "DASHSCOPE_API_BASE"},
    "togetherai": {"api_key": "TOGETHER_API_KEY", "api_base": "TOGETHER_BASE_URL"},
    "ark": {"api_key": "ARK_API_KEY", "api_base": "ARK_BASE_URL"},
    "qianfan": {"api_key": "QIANFAN_API_KEY", "api_base": "QIANFAN_BASE_URL"},
    "lingyiwanwu": {"api_key": "LINGYIWANWU_API_KEY", "api_base": "LINGYIWANWU_BASE_URL"},
}


# Safe defaults (only used when env/file is empty).
_DEFAULT_API_BASE: dict[str, str] = {
    "siliconflow": "https://api.siliconflow.cn/v1",
    "openai": "https://api.openai.com/v1",
    "deepseek": "https://api.deepseek.com/v1",
    # NOTE: some providers use non-standard paths; users can override via UI.
    "zhipu": "https://open.bigmodel.cn/api/paas/v4",
    "openrouterai": "https://openrouter.ai/api/v1",
    "dashscope": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "togetherai": "https://api.together.xyz/v1",
    # OpenAI-compatible endpoints (can be overridden in Settings UI).
    "ark": "https://ark.cn-beijing.volces.com/api/v3",
    "qianfan": "https://qianfan.baidubce.com/v2",
    "lingyiwanwu": "https://api.01.ai/v1",
}


def _config_dir() -> Path:
    return Path(settings.paths.save_yaml_path) / "config"


def _secrets_file() -> Path:
    return _config_dir() / "provider_secrets.json"


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


def mask_secret(value: str, keep_start: int = 3, keep_end: int = 4) -> str:
    value = (value or "").strip()
    if not value:
        return ""
    if len(value) <= keep_start + keep_end:
        return "*" * len(value)
    return value[:keep_start] + ("*" * (len(value) - keep_start - keep_end)) + value[-keep_end:]


@lru_cache(maxsize=1)
def _load_provider_secrets_cached() -> Dict[str, Any]:
    return _read_json_file(_secrets_file())


def load_provider_secrets() -> Dict[str, Any]:
    """Load persisted provider secrets (API keys / base URLs)."""
    with _lock:
        return deepcopy(_load_provider_secrets_cached())


def _provider_env_key(provider: str) -> Optional[str]:
    p = _canonical_provider(provider)
    return _PROVIDER_ENV.get(p, {}).get("api_key")


def _provider_env_base(provider: str) -> Optional[str]:
    p = _canonical_provider(provider)
    return _PROVIDER_ENV.get(p, {}).get("api_base")


def get_provider_api_key(provider: str) -> str:
    raw = (provider or "").strip().lower()
    provider = _canonical_provider(raw)
    secrets = load_provider_secrets()
    key = (secrets.get(provider, {}) or {}).get("api_key") or ""
    if not key and raw and raw != provider:
        # Backward-compat for any previously stored alias key.
        key = (secrets.get(raw, {}) or {}).get("api_key") or ""
    if not key:
        # Also check known aliases pointing to this canonical provider.
        for alias, canon in _ALIAS_TO_CANONICAL.items():
            if canon != provider:
                continue
            key = (secrets.get(alias, {}) or {}).get("api_key") or ""
            if key:
                break
    if key:
        return str(key).strip()

    env_key = _provider_env_key(provider)
    if env_key:
        return (os.getenv(env_key) or "").strip()

    return ""


def get_provider_api_base(provider: str) -> str:
    raw = (provider or "").strip().lower()
    provider = _canonical_provider(raw)
    secrets = load_provider_secrets()
    base = (secrets.get(provider, {}) or {}).get("api_base") or ""
    if not base and raw and raw != provider:
        base = (secrets.get(raw, {}) or {}).get("api_base") or ""
    if not base:
        for alias, canon in _ALIAS_TO_CANONICAL.items():
            if canon != provider:
                continue
            base = (secrets.get(alias, {}) or {}).get("api_base") or ""
            if base:
                break
    if base:
        return str(base).strip().rstrip("/")

    env_base = _provider_env_base(provider)
    if env_base:
        val = (os.getenv(env_base) or "").strip()
        if val:
            return val.rstrip("/")

    # Fallback to known defaults, then to global llm base.
    return (_DEFAULT_API_BASE.get(provider) or settings.llm.api_base or "").strip().rstrip("/")


def patch_provider_secrets(provider: str, api_key: Optional[str] = None, api_base: Optional[str] = None) -> Dict[str, Any]:
    """
    Patch a single provider secret.

    - Empty string clears the stored value.
    - We DO NOT return the raw secret value (caller should use `build_provider_status()`).
    """
    raw = (provider or "").strip().lower()
    provider = _canonical_provider(raw)
    if not provider:
        raise ValueError("provider is required")

    with _lock:
        cur = _read_json_file(_secrets_file())
        # If this provider was previously stored under an alias, migrate to canonical key.
        for alias, canon in _ALIAS_TO_CANONICAL.items():
            if canon == provider:
                cur.pop(alias, None)
        provider_cfg = dict(cur.get(provider, {}) or {})

        if api_key is not None:
            api_key = (api_key or "").strip()
            if api_key:
                provider_cfg["api_key"] = api_key
            else:
                provider_cfg.pop("api_key", None)

        if api_base is not None:
            api_base = (api_base or "").strip()
            if api_base:
                provider_cfg["api_base"] = api_base.rstrip("/")
            else:
                provider_cfg.pop("api_base", None)

        if provider_cfg:
            cur[provider] = provider_cfg
        else:
            cur.pop(provider, None)

        _atomic_write_json(_secrets_file(), cur)
        _load_provider_secrets_cached.cache_clear()

    # Best-effort: update process env so other modules using os.getenv can see it immediately.
    env_key = _provider_env_key(provider)
    if api_key is not None and env_key:
        if api_key:
            os.environ[env_key] = api_key
        else:
            os.environ.pop(env_key, None)

    env_base = _provider_env_base(provider)
    if api_base is not None and env_base:
        if api_base:
            os.environ[env_base] = api_base.rstrip("/")
        else:
            os.environ.pop(env_base, None)

    return load_provider_secrets()


def patch_provider_secrets_many(patch: Dict[str, Any]) -> Dict[str, Any]:
    """
    Patch multiple providers at once.

    Accepts:
      {"openai": {"api_key": "...", "api_base": "..."}, "deepseek": {...}}
    """
    patch = patch or {}
    normalized: dict[str, dict[str, Any]] = {}
    for k, v in patch.items():
        if not k:
            continue
        if not isinstance(v, dict):
            continue
        provider = _canonical_provider(k)
        if not provider:
            continue
        # Allow aliases in payload while keeping storage canonical.
        merged = dict(normalized.get(provider, {}) or {})
        merged.update(v)
        normalized[provider] = merged

    if not normalized:
        return load_provider_secrets()

    with _lock:
        cur = _read_json_file(_secrets_file())
        for provider, cfg in normalized.items():
            provider = _canonical_provider(provider)
            if not provider:
                continue
            for alias, canon in _ALIAS_TO_CANONICAL.items():
                if canon == provider:
                    cur.pop(alias, None)
            provider_cfg = dict(cur.get(provider, {}) or {})

            if "api_key" in cfg:
                key = (cfg.get("api_key") or "").strip()
                if key:
                    provider_cfg["api_key"] = key
                else:
                    provider_cfg.pop("api_key", None)

            if "api_base" in cfg:
                base = (cfg.get("api_base") or "").strip()
                if base:
                    provider_cfg["api_base"] = base.rstrip("/")
                else:
                    provider_cfg.pop("api_base", None)

            if provider_cfg:
                cur[provider] = provider_cfg
            else:
                cur.pop(provider, None)

        _atomic_write_json(_secrets_file(), cur)
        _load_provider_secrets_cached.cache_clear()

    # Best-effort: update env for known providers.
    for provider, cfg in normalized.items():
        provider = _canonical_provider(provider)
        if not provider:
            continue
        if "api_key" in cfg:
            env_key = _provider_env_key(provider)
            if env_key:
                key = (cfg.get("api_key") or "").strip()
                if key:
                    os.environ[env_key] = key
                else:
                    os.environ.pop(env_key, None)
        if "api_base" in cfg:
            env_base = _provider_env_base(provider)
            if env_base:
                base = (cfg.get("api_base") or "").strip().rstrip("/")
                if base:
                    os.environ[env_base] = base
                else:
                    os.environ.pop(env_base, None)

    return load_provider_secrets()


def build_provider_status() -> Dict[str, Any]:
    """
    Return UI-safe provider status for settings page.
    IMPORTANT: never include raw API keys in the response.
    """
    secrets = load_provider_secrets()
    providers = set(_DEFAULT_API_BASE.keys())
    for k in (secrets or {}).keys():
        providers.add(_canonical_provider(k))

    def _merged_stored(provider: str) -> Dict[str, Any]:
        merged: Dict[str, Any] = dict(secrets.get(provider, {}) or {})
        # Backward-compat: merge any existing alias entries into the canonical provider view.
        for alias, canon in _ALIAS_TO_CANONICAL.items():
            if canon != provider:
                continue
            a = secrets.get(alias, {}) or {}
            if not merged.get("api_key") and a.get("api_key"):
                merged["api_key"] = a.get("api_key")
            if not merged.get("api_base") and a.get("api_base"):
                merged["api_base"] = a.get("api_base")
        return merged

    status: Dict[str, Any] = {}
    for provider in sorted(providers):
        provider = _canonical_provider(provider)
        if not provider:
            continue

        stored = _merged_stored(provider)
        stored_key = (stored.get("api_key") or "").strip()
        stored_base = (stored.get("api_base") or "").strip()

        env_key_name = _provider_env_key(provider)
        env_base_name = _provider_env_base(provider)
        env_key = (os.getenv(env_key_name) or "").strip() if env_key_name else ""
        env_base = (os.getenv(env_base_name) or "").strip() if env_base_name else ""

        api_key = stored_key or env_key
        api_base = stored_base or env_base or _DEFAULT_API_BASE.get(provider) or settings.llm.api_base

        status[provider] = {
            "api_base": (api_base or "").strip().rstrip("/"),
            "configured": bool(api_key),
            "api_key_masked": mask_secret(api_key) if api_key else "",
            "source": "file" if (stored_key or stored_base) else ("env" if (env_key or env_base) else "default"),
        }

    return status
