from abc import ABC, abstractmethod
from typing import Any, Dict, AsyncIterator, Optional, List
from langchain_core.messages import BaseMessage
from langgraph.graph.state import CompiledStateGraph

class BaseAgent(ABC):
    """
    Agent 基类
    
    所有 Agent 实现都应继承此类，并提供统一的接口用于：
    - 消息流式处理 (stream_messages)
    - 状态查询与更新 (get_state, update_state)
    - 元数据获取 (get_info)
    """

    @property
    @abstractmethod
    def graph(self) -> CompiledStateGraph:
        """返回编译后的 LangGraph 图实例"""
        pass

    @abstractmethod
    def get_info(self) -> dict:
        """返回 Agent 元数据"""
        pass

    async def ainvoke(self, input: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """异步调用 Agent"""
        return await self.graph.ainvoke(input, config)

    async def astream(self, input: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> AsyncIterator[Any]:
        """异步流式调用 Agent"""
        async for chunk in self.graph.astream(input, config):
            yield chunk

    async def get_state(self, thread_id: str):
        """获取指定线程的状态"""
        config = {"configurable": {"thread_id": thread_id}}
        return await self.graph.aget_state(config)

    async def update_state(self, thread_id: str, values: Dict[str, Any], as_node: Optional[str] = None):
        """更新指定线程的状态"""
        config = {"configurable": {"thread_id": thread_id}}
        return await self.graph.aupdate_state(config, values, as_node=as_node)

    async def get_state_history(self, thread_id: str, limit: int = 10):
        """获取状态历史"""
        config = {"configurable": {"thread_id": thread_id}}
        # CompiledGraph.aget_state_history returns an iterator
        history = []
        async for state in self.graph.aget_state_history(config, limit=limit):
            history.append(state)
        return history
