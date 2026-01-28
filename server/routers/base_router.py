from fastapi import APIRouter, Body, HTTPException

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
        raise HTTPException(status_code=500, detail=str(e)) from e


@base.post("/config")
async def update_config(key: str = Body(...), value=Body(...)):
    """
    Backward-compatible single-key update used by older frontend code.
    """
    try:
        patch_ui_overrides({key: value})
        return build_ui_config()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@base.post("/restart")
async def restart():
    # Clear cached runtime singletons so subsequent requests rebuild clients/models.
    # This avoids instantiating optional services (Milvus/Neo4j/etc) when the feature is disabled.
    from src.runtime import reset_all

    reset_all()
    try:
        from src.graph.runtime import reset_graph_workers

        reset_graph_workers()
    except Exception:
        pass
    return {"message": "Restarted!"}
