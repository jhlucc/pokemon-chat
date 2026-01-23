from fastapi import APIRouter
from server.routers.chat_router import chat
from server.routers.data_router import data
from server.routers.base_router import base
from server.routers.tool_router import router as tool
from server.routers.admin_router import admin
from server.routers.mcp_router import router as mcp
from server.routers.agent_router import router as agent, agents_router
from server.routers.health_router import health
from server.routers.refresh_router import refresh_router
from server.routers.memory_router import router as memory_router
from server.routers.log_router import router as log_router

router = APIRouter()
router.include_router(base)
router.include_router(health)
router.include_router(chat)
router.include_router(data)
router.include_router(tool)
router.include_router(agent)
router.include_router(agents_router)  # /api/agents/
router.include_router(admin)
router.include_router(mcp)
router.include_router(refresh_router)
router.include_router(memory_router)
router.include_router(log_router)
