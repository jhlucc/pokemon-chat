from fastapi import APIRouter, HTTPException, BackgroundTasks, Body
from typing import Dict, Any, List, Optional
from src.agents.manager import agent_manager
from src.utils.logger import LogManager

logger = LogManager()
router = APIRouter(prefix="/agent")

@router.get("/{agent_name}/state/{thread_id}")
async def get_agent_state(agent_name: str, thread_id: str):
    """获取 Agent 的当前状态"""
    try:
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
                "parent_config": state.parent_config
            }
        else:
            raise HTTPException(status_code=400, detail="该 Agent 不支持状态查询")
    except Exception as e:
        logger.error(f"Error getting state: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{agent_name}/history/{thread_id}")
async def get_agent_history(agent_name: str, thread_id: str, limit: int = 10):
    """获取 Agent 的状态历史（时间旅行）"""
    try:
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
                        "parent_config": s.parent_config
                    }
                    for s in history
                ]
            }
        else:
            raise HTTPException(status_code=400, detail="该 Agent 不支持历史查询")
    except Exception as e:
        logger.error(f"Error getting history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{agent_name}/state/{thread_id}")
async def update_agent_state(
    agent_name: str,
    thread_id: str,
    payload: Dict[str, Any] = Body(...)
):
    """更新 Agent 状态（干预/回滚）"""
    try:
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
        raise HTTPException(status_code=500, detail=str(e))
