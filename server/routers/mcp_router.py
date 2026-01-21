# api_router.py
import json
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query, FastAPI
from pydantic import BaseModel
from contextlib import asynccontextmanager
from src.runtime import get_mcp_client

router = APIRouter(prefix="/mcp", tags=["mcp"])

# ---------- models ----------
class ChatResp(BaseModel):
    answer: str

class CoordsResp(BaseModel):
    coords: List[dict]

# ---------- routes ----------
@router.get("/chat", response_model=ChatResp)
async def chat(q: str = Query(..., description="聊天提问")):
    try:
        client = get_mcp_client()
        answer, _ = await client.ask(q)
    except Exception as e:
        raise HTTPException(500, detail=f"MCP/LLM 调用失败: {e}")


    return ChatResp(answer=answer)

@router.get("/coords", response_model=CoordsResp)
async def coords(place: str = Query(..., description="地点名")):
    try:
        client = get_mcp_client()
        _, coords_json = await client.ask(f"{place} 出现在真实世界的具体坐标")
    except Exception as e:
        raise HTTPException(500, detail=f"MCP/LLM 调用失败: {e}")

    if not coords_json:
        raise HTTPException(404, detail="未查询到坐标")

    # 过滤掉 lat 或 lng 为 null 的项
    coords_list = [c for c in json.loads(coords_json)
                   if c.get("lat") is not None and c.get("lng") is not None]

    if not coords_list:
        raise HTTPException(404, detail="无可用坐标")

    return CoordsResp(coords=coords_list)

# ---------- lifespan ----------
@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    await get_mcp_client().aclose()
