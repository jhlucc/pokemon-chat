from fastapi import APIRouter

from src.core.settings import settings
from src.utils.logger import get_logger

_log = get_logger(__name__)

router = APIRouter()


# Core routers (always enabled)
from server.routers.base_router import base  # noqa: E402
from server.routers.health_router import health  # noqa: E402
from server.routers.provider_router import router as provider_router  # noqa: E402
from server.routers.chat_router import chat  # noqa: E402
from server.routers.data_router import data  # noqa: E402
from server.routers.tool_router import router as tool  # noqa: E402
from server.routers.agent_router import router as agent, agents_router  # noqa: E402
from server.routers.admin_router import admin  # noqa: E402
from server.routers.log_router import router as log_router  # noqa: E402

router.include_router(base)
router.include_router(health)
router.include_router(provider_router)
router.include_router(chat)
router.include_router(data)
router.include_router(tool)
router.include_router(agent)
router.include_router(agents_router)  # /api/agents/
router.include_router(admin)
router.include_router(log_router)


# Optional routers are always registered, but endpoints will self-check feature flags.
# This allows toggling backend capabilities from the Settings UI without restarting the server.
try:
    from server.routers.mcp_router import router as mcp_router  # noqa: E402

    router.include_router(mcp_router)
except Exception as e:
    _log.warning(f"MCP router disabled (import failed): {e}")

try:
    from server.routers.refresh_router import refresh_router  # noqa: E402

    router.include_router(refresh_router)
except Exception as e:
    _log.warning(f"Knowledge refresh router disabled (import failed): {e}")

# Memory is best-effort: skip if optional deps are missing.
try:
    from server.routers.memory_router import router as memory_router  # noqa: E402

    router.include_router(memory_router)
except Exception as e:
    _log.warning(f"Memory router disabled (import failed): {e}")
