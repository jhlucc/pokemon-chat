"""
MCP Worker - Agent Node for MySQL/地理位置查询

Integrates MCP (Model Context Protocol) tools into the LangGraph agent.
Used for querying Pokemon location data from MySQL.
"""

from typing import Any

from langchain_core.messages import AIMessage

from src.core.feature_flags import feature_enabled
from src.graph.state import AgentState
from src.utils.logger import get_logger

logger = get_logger(__name__)


class MCPWorker:
    """
    MCP Worker node for geographic/location queries.

    Handles queries like:
    - "皮卡丘出现在哪里？"
    - "真新镇的真实坐标?"
    """

    def __init__(self):
        # Keep init cheap; the feature can be toggled at runtime.
        pass

    async def _query_mcp(self, query: str) -> dict[str, Any]:
        """Execute MCP query with caching."""
        if not feature_enabled("enable_mcp"):
            return {"answer": "", "coords": None}

        try:
            from src.mcp.client_core import cached_ask

            answer, coords = await cached_ask(query)
            return {"answer": answer or "", "coords": coords}
        except Exception as e:
            logger.error(f"MCP query failed: {e}")
            return {"answer": f"地理查询失败: {e}", "coords": None}

    async def __call__(self, state: AgentState) -> dict[str, Any]:
        """
        Worker node entry point.
        """

        messages = state["messages"]
        last_message = messages[-1]
        query = last_message.content

        logger.info(f"MCPWorker processing: {query[:50]}...")

        # Execute MCP query
        result = await self._query_mcp(query)

        if result["answer"]:
            response_text = f"根据地理数据库查询结果:\n\n{result['answer']}"

            # If we have coordinates, format them nicely
            if result["coords"]:
                try:
                    import json

                    coords_data = json.loads(result["coords"])
                    if coords_data and isinstance(coords_data, list):
                        locations = "\n".join(
                            [
                                f"- {item.get('location', '未知')}: ({item.get('lat', 0):.4f}, {item.get('lng', 0):.4f})"
                                for item in coords_data
                            ]
                        )
                        response_text += f"\n\n📍 坐标信息:\n{locations}"
                except Exception as e:
                    logger.debug(f"Failed to parse MCP coords payload: {e}")
        else:
            response_text = "抱歉，未能查询到相关地理位置信息。"

        return {"messages": [AIMessage(content=response_text)]}


# Global instance
_mcp_worker: MCPWorker = None


def get_mcp_worker() -> MCPWorker:
    global _mcp_worker
    if _mcp_worker is None:
        _mcp_worker = MCPWorker()
    return _mcp_worker
