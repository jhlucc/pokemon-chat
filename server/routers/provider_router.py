from __future__ import annotations

from fastapi import APIRouter, Body, HTTPException

from src.core.provider_config import (
    build_provider_status,
    patch_provider_secrets,
    patch_provider_secrets_many,
)

router = APIRouter(tags=["providers"])


@router.get("/providers")
def get_providers():
    return {"providers": build_provider_status()}


@router.patch("/providers")
def patch_providers(payload: dict = Body(...)):
    """
    Update provider credentials/base URLs.

    Accepts:
    - Single provider:
        {"provider": "openai", "api_key": "...", "api_base": "..."}
    - Multiple providers:
        {"openai": {"api_key": "...", "api_base": "..."}, "deepseek": {...}}
    """
    try:
        if isinstance(payload, dict) and isinstance(payload.get("provider"), str):
            provider = payload.get("provider")
            api_key = payload.get("api_key", None)
            api_base = payload.get("api_base", None)
            patch_provider_secrets(provider=provider, api_key=api_key, api_base=api_base)
        else:
            patch_provider_secrets_many(payload)
        return {"providers": build_provider_status()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

