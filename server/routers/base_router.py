from fastapi import Body, APIRouter, HTTPException

# from src import config, knowledge_base
from src.core.settings import settings
from src.runtime import get_retriever, get_kb
from server.runtime_config import build_ui_config, patch_ui_overrides

base = APIRouter()

@base.get("/")
async def route_index():
    return {"message": "You Got It!"}

@base.get("/config")
def get_config():
    # Never return secrets (API keys). Only return UI-safe config.
    return build_ui_config()

@base.patch("/config")
async def patch_config(patch: dict = Body(...)):
    try:
        patch_ui_overrides(patch)
        return build_ui_config()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@base.post("/config")
async def update_config(key: str = Body(...), value=Body(...)):
    """
    Backward-compatible single-key update used by older frontend code.
    """
    try:
        patch_ui_overrides({key: value})
        return build_ui_config()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@base.post("/restart")
async def restart():
    get_kb().restart()
    get_retriever().restart()
    return {"message": "Restarted!"}


