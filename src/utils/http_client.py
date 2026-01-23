from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional

import httpx


def _has_unsupported_proxy_scheme() -> bool:
    """
    The runtime/test environment may set SOCKS proxies (e.g. socks://127.0.0.1:xxxx).
    `httpx` (without extra deps) will raise on these schemes.
    """
    for key in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    ):
        v = os.getenv(key)
        if not v:
            continue
        if v.lower().startswith("socks"):
            return True
    return False


def _strip_unsupported_proxy_env() -> None:
    """
    LangChain/OpenAI stacks may read proxy env vars and validate schemes.
    If the environment provides SOCKS proxies (unsupported without extras),
    we proactively remove those variables to keep the app/test environment usable.
    """
    for key in ("ALL_PROXY", "all_proxy", "OPENAI_PROXY", "openai_proxy"):
        v = os.getenv(key)
        if v and v.lower().startswith("socks"):
            os.environ.pop(key, None)


@lru_cache(maxsize=1)
def get_safe_httpx_client() -> Optional[httpx.Client]:
    """
    Return a shared httpx client that avoids env proxy parsing issues.
    If the environment doesn't use unsupported proxy schemes, return None
    so callers can use the default client behavior.
    """
    if _has_unsupported_proxy_scheme():
        _strip_unsupported_proxy_env()
        # Do not trust env proxies when they are not supported by httpx.
        return httpx.Client(trust_env=False, timeout=httpx.Timeout(30.0))
    return None
