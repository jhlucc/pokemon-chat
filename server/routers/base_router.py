from fastapi import Body, APIRouter

# from src import config, knowledge_base
from src.core.settings import settings
from src.runtime import get_retriever, get_kb

base = APIRouter()

@base.get("/")
async def route_index():
    return {"message": "You Got It!"}

@base.get("/config")
def get_config():
    return settings.model_dump()

@base.post("/config")
async def update_config(key = Body(...), value = Body(...)):
    # if key == "custom_models":
    #     value = config.compare_custom_models(value)
    # config[key] = value
    # config.save()
    # return config.get_safe_config()
    from fastapi import HTTPException
    raise HTTPException(status_code=501, detail="Configuration update not supported")

@base.post("/restart")
async def restart():
    get_kb().restart()
    get_retriever().restart()
    return {"message": "Restarted!"}



