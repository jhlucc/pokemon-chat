import logging
from threading import Lock
from typing import List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from mem0 import Memory
import os

from src.core.settings import settings

router = APIRouter(prefix="/memory", tags=["memory"])
logger = logging.getLogger(__name__)

# Initialize Mem0 lazily to avoid heavy side effects at import time (e.g. Qdrant local init).
_memory_client: Memory | None = None
_lock = Lock()


def get_memory_client() -> Memory:
    global _memory_client
    if _memory_client is not None:
        return _memory_client

    with _lock:
        if _memory_client is not None:
            return _memory_client

        # Using basic config for now. For production, configure Qdrant/Milvus explicitly.
        # Do not clobber user-provided env values.
        os.environ.setdefault("OPENAI_API_KEY", settings.get_api_key("openai") or "")
        if settings.llm.api_base:
            os.environ.setdefault("OPENAI_BASE_URL", settings.llm.api_base)

        _memory_client = Memory()
        return _memory_client

class MemoryAddRequest(BaseModel):
    user_id: str
    text: str
    metadata: Optional[dict] = None

class MemorySearchRequest(BaseModel):
    user_id: str
    query: str
    limit: int = 5

@router.post("/add")
async def add_memory(request: MemoryAddRequest):
    try:
        result = get_memory_client().add(request.text, user_id=request.user_id, metadata=request.metadata)
        return {"status": "success", "result": result}
    except Exception as e:
        logger.error(f"Error adding memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/all")
async def get_all_memories(user_id: str):
    try:
        if not user_id:
             raise HTTPException(status_code=400, detail="user_id required")
        memories = get_memory_client().get_all(user_id=user_id)
        return {"result": memories}
    except Exception as e:
        logger.error(f"Error getting memories: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/search")
async def search_memory(request: MemorySearchRequest):
    try:
        memories = get_memory_client().search(request.query, user_id=request.user_id, limit=request.limit)
        return {"result": memories}
    except Exception as e:
        logger.error(f"Error searching memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/")
async def delete_memory(memory_id: str):
    try:
        get_memory_client().delete(memory_id)
        return {"status": "success"}
    except Exception as e:
        logger.error(f"Error deleting memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/all")
async def delete_all_memories(user_id: str):
    try:
        get_memory_client().delete_all(user_id=user_id)
        return {"status": "success"}
    except Exception as e:
        logger.error(f"Error deleting all memories: {e}")
        raise HTTPException(status_code=500, detail=str(e))
