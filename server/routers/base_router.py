from fastapi import Request, Body
from fastapi import APIRouter

base = APIRouter()
from src import config, get_retriever, knowledge_base
from src.agents.kg_agent import KGQueryAgent
kg_agent = KGQueryAgent()        # Neo4j 图谱检索器
retriever = get_retriever()

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
    retriever.restart()
    return {"message": "Restarted!"}



