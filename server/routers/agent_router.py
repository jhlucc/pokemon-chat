from typing import Any

from fastapi import APIRouter, Body, HTTPException

from src.utils.logger import LogManager

logger = LogManager()
router = APIRouter(prefix="/agent")

# 根路由需要单独，这里添加一个 /agents 路由用于前端
agents_router = APIRouter(prefix="/agents")


@agents_router.get("/")
async def list_agents():
    """列出所有可用的 Agents"""
    try:
        from src.agents.manager import agent_manager

        agents = agent_manager.list_agents()
        return {
            "agents": [
                {
                    "name": name,
                    "description": info.get("description", ""),
                    "supports_streaming": info.get("supports_streaming", True),
                }
                for name, info in agents.items()
            ]
        }
    except Exception as e:
        logger.error(f"Error listing agents: {e}")
        # 返回默认的 supervisor_agent
        return {
            "agents": [{"name": "supervisor_agent", "description": "智能多工具协调Agent", "supports_streaming": True}]
        }


@router.get("/{agent_name}/state/{thread_id}")
async def get_agent_state(agent_name: str, thread_id: str):
    """获取 Agent 的当前状态"""
    try:
        from src.agents.manager import agent_manager

        agent = agent_manager.get_agent(agent_name)
        if hasattr(agent, "get_state"):
            state = await agent.get_state(thread_id)
            # 序列化 State
            return {
                "values": state.values,
                "next": state.next,
                "metadata": state.metadata,
                "created_at": state.created_at,
                "config": state.config,
                "parent_config": state.parent_config,
            }
        else:
            raise HTTPException(status_code=400, detail="该 Agent 不支持状态查询")
    except Exception as e:
        logger.error(f"Error getting state: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/{agent_name}/history/{thread_id}")
async def get_agent_history(agent_name: str, thread_id: str, limit: int = 10):
    """获取 Agent 的状态历史（时间旅行）"""
    try:
        from src.agents.manager import agent_manager

        agent = agent_manager.get_agent(agent_name)
        if hasattr(agent, "get_state_history"):
            history = await agent.get_state_history(thread_id, limit=limit)
            return {
                "history": [
                    {
                        "values": s.values,
                        "next": s.next,
                        "metadata": s.metadata,
                        "created_at": s.created_at,
                        "config": s.config,
                        "parent_config": s.parent_config,
                    }
                    for s in history
                ]
            }
        else:
            raise HTTPException(status_code=400, detail="该 Agent 不支持历史查询")
    except Exception as e:
        logger.error(f"Error getting history: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/{agent_name}/state/{thread_id}")
async def update_agent_state(agent_name: str, thread_id: str, payload: dict[str, Any] = Body(...)):
    """更新 Agent 状态（干预/回滚）"""
    try:
        from src.agents.manager import agent_manager

        agent = agent_manager.get_agent(agent_name)
        if hasattr(agent, "update_state"):
            values = payload.get("values", {})
            as_node = payload.get("as_node")

            result = await agent.update_state(thread_id, values, as_node=as_node)
            return {"status": "success", "config": result}
        else:
            raise HTTPException(status_code=400, detail="该 Agent 不支持状态更新")
    except Exception as e:
        logger.error(f"Error updating state: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/{agent_name}/resume/{thread_id}")
async def resume_agent(agent_name: str, thread_id: str, payload: dict[str, Any] = Body(...)):
    """恢复被中断的 Agent 执行"""
    try:
        # Lazy import: keep server startup cheap.
        from langgraph.types import Command

        from src.agents.manager import agent_manager

        agent = agent_manager.get_agent(agent_name)
        if not hasattr(agent, "graph"):
            raise HTTPException(status_code=400, detail="该 Agent 不支持图执行")

        # 使用 Command(resume=...) 恢复
        # payload 通常包含用户对 interrupt 的反馈
        resume_value = payload.get("resume")
        if resume_value is None:
            # 如果 payload 自身就是 resume 值
            resume_value = payload

        config = {"configurable": {"thread_id": thread_id}}

        # 异步调用 invoke 继续执行
        # 注意: 这里 invoke 会继续运行直到结束或下一个中断
        # 我们可能希望 background 运行或者 stream，为了简单这里 await 结果
        # 但通常 resume 后是继续之前的 stream 逻辑
        # 这里简单 await 返回最终状态
        await agent.graph.ainvoke(Command(resume=resume_value), config)

        return {"status": "resumed"}

    except Exception as e:
        logger.error(f"Error resuming agent: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e
