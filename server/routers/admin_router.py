from __future__ import annotations

import os
import secrets
import string
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel

from server.db_manager import db_manager


def _extract_admin_token(request: Request) -> str:
    # Prefer a dedicated header, but also accept Authorization: Bearer <token>.
    raw = (request.headers.get("X-Admin-Token") or request.headers.get("x-admin-token") or "").strip()
    if raw:
        return raw
    auth = (request.headers.get("Authorization") or "").strip()
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return ""


def require_admin(request: Request) -> None:
    """
    Optional admin protection.

    - If ADMIN_API_KEY is empty -> admin endpoints remain open (backward compatible).
    - If ADMIN_API_KEY is set  -> require X-Admin-Token (or Authorization: Bearer ...) to match.
    """
    expected = (os.getenv("ADMIN_API_KEY") or "").strip()
    if not expected:
        return
    provided = _extract_admin_token(request)
    if not provided or not secrets.compare_digest(provided, expected):
        raise HTTPException(status_code=401, detail="Unauthorized")


# 把所有后台管理接口挂在 /admin/*
admin = APIRouter(prefix="/admin", tags=["admin"], dependencies=[Depends(require_admin)])


# 依赖项：获取数据库会话
def get_db():
    db = db_manager.get_session()
    try:
        yield db
    finally:
        db.close()


# 请求和响应模型
class TokenCreate(BaseModel):
    agent_id: str
    name: str


class TokenVerify(BaseModel):
    agent_id: str
    token: str


class TokenResponse(BaseModel):
    id: int
    agent_id: str
    name: str
    token: str
    created_at: str


# 生成随机token
def generate_token(length=32):
    alphabet = string.ascii_letters + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(length))


@admin.get("/tokens", response_model=list[TokenResponse])
def get_agent_tokens(agent_id: str | None = Query(None), db: Any = Depends(get_db)):
    """获取智能体的token列表"""
    from src.models.token_model import AgentToken

    query = db.query(AgentToken)
    if agent_id:
        query = query.filter(AgentToken.agent_id == agent_id)
    tokens = query.all()
    return [token.to_dict() for token in tokens]


@admin.post("/tokens", response_model=TokenResponse)
def create_token(token_data: TokenCreate, db: Any = Depends(get_db)):
    """创建新的token"""
    from src.models.token_model import AgentToken

    # 生成随机token
    token_value = generate_token()

    # 创建token记录
    new_token = AgentToken(agent_id=token_data.agent_id, name=token_data.name, token=token_value)

    # 保存到数据库
    db.add(new_token)
    db.commit()
    db.refresh(new_token)

    return new_token.to_dict()


@admin.delete("/tokens/{token_id}", response_model=dict)
def delete_token(token_id: int, db: Any = Depends(get_db)):
    """删除token"""
    from src.models.token_model import AgentToken

    token = db.query(AgentToken).filter(AgentToken.id == token_id).first()
    if not token:
        raise HTTPException(status_code=404, detail="Token not found")

    db.delete(token)
    db.commit()

    return {"success": True, "message": "Token deleted"}


@admin.post("/verify_token")
def verify_agent_token(token_data: TokenVerify, db: Any = Depends(get_db)):
    """验证智能体访问令牌"""
    from src.models.token_model import AgentToken

    token = (
        db.query(AgentToken)
        .filter(AgentToken.agent_id == token_data.agent_id, AgentToken.token == token_data.token)
        .first()
    )

    if not token:
        raise HTTPException(status_code=401, detail="Invalid token")

    return {"success": True, "message": "Token verified"}


# ---------------------------------------------------------------------------
# Cache admin endpoints (best-effort, no auth in this demo project)
# ---------------------------------------------------------------------------


@admin.get("/cache/semantic")
def get_semantic_cache_stats():
    """Inspect semantic cache status (size/ttl/threshold)."""
    from src.knowledge.cache.cache import get_semantic_cache

    cache = get_semantic_cache()
    return {"cache": cache.stats()}


@admin.post("/cache/semantic/clear")
def clear_semantic_cache():
    """Clear semantic cache entries."""
    from src.knowledge.cache.cache import get_semantic_cache

    cache = get_semantic_cache()
    cache.clear()
    return {"success": True, "cache": cache.stats()}
