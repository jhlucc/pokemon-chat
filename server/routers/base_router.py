from fastapi import Body, APIRouter

from src import config, knowledge_base
from src.runtime import get_retriever

base = APIRouter()

@base.get("/")
async def route_index():
    return {"message": "You Got It!"}

@base.get("/config")
def get_config():
    return config.get_safe_config()

@base.post("/config")
async def update_config(key = Body(...), value = Body(...)):
    if key == "custom_models":
        value = config.compare_custom_models(value)

    config[key] = value
    config.save()
    return config.get_safe_config()

@base.post("/restart")
async def restart():
    knowledge_base.restart()
    get_retriever().restart()
    return {"message": "Restarted!"}



