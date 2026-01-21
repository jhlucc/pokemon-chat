from abc import ABC, abstractmethod
from typing import Any, Dict, AsyncIterator, Optional, List, TypeVar, Generic, Type
from langchain_core.messages import BaseMessage
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.base import BaseCheckpointSaver

from src.core.settings import settings

# Type variable for state schema
TState = TypeVar('TState', bound=Dict[str, Any])


class BaseAgent(ABC, Generic[TState]):
    """
    Agent 基类
    
    所有 Agent 实现都应继承此类，并提供统一的接口用于：
    - 消息流式处理 (astream, ainvoke)
    - 状态查询与更新 (get_state, update_state)
    - 元数据获取 (get_info)
    
    子类必须实现:
    - _build_graph(): 构建并返回编译后的 LangGraph 图
    - get_info(): 返回 Agent 元数据
    """

    def __init__(
        self,
        llm: Optional[BaseChatModel] = None,
        tools: Optional[List[BaseTool]] = None,
        checkpointer: Optional[BaseCheckpointSaver] = None,
        **kwargs
    ):
        """
        初始化基础 Agent
        
        Args:
            llm: 语言模型实例，默认从 settings 创建
            tools: 工具列表
            checkpointer: 状态检查点保存器，默认使用 MemorySaver
            **kwargs: 额外参数传递给子类
        """
        self._llm = llm
        self._tools = tools or []
        self._checkpointer = checkpointer or MemorySaver()
        self._graph: Optional[CompiledStateGraph] = None
        
        # 调用子类初始化钩子
        self._init_components(**kwargs)
        
        # 构建图
        self._graph = self._build_graph()

    @property
    def llm(self) -> BaseChatModel:
        """获取语言模型实例，延迟初始化"""
        if self._llm is None:
            self._llm = self._default_llm()
        return self._llm

    @property
    def tools(self) -> List[BaseTool]:
        """获取工具列表"""
        return self._tools

    @property
    def checkpointer(self) -> BaseCheckpointSaver:
        """获取检查点保存器"""
        return self._checkpointer

    @property
    def graph(self) -> CompiledStateGraph:
        """返回编译后的 LangGraph 图实例"""
        if self._graph is None:
            self._graph = self._build_graph()
        return self._graph

    def _default_llm(self) -> BaseChatModel:
        """
        创建默认的 LLM 实例
        
        子类可以覆盖此方法以使用不同的模型配置
        """
        return ChatOpenAI(
            model=settings.llm.model_name,
            api_key=settings.llm.api_key,
            base_url=settings.llm.api_base,
            temperature=0.7,
        )

    def _init_components(self, **kwargs) -> None:
        """
        初始化组件钩子
        
        子类可以覆盖此方法来初始化额外的组件
        """
        pass

    @abstractmethod
    def _build_graph(self) -> CompiledStateGraph:
        """
        构建并返回编译后的 LangGraph 图
        
        子类必须实现此方法
        """
        pass

    @abstractmethod
    def get_info(self) -> dict:
        """返回 Agent 元数据"""
        pass

    # ==================== 调用接口 ====================

    async def ainvoke(self, input: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """异步调用 Agent"""
        return await self.graph.ainvoke(input, config)

    def invoke(self, input: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """同步调用 Agent"""
        return self.graph.invoke(input, config)

    async def astream(self, input: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> AsyncIterator[Any]:
        """异步流式调用 Agent"""
        async for chunk in self.graph.astream(input, config):
            yield chunk

    def stream(self, input: Dict[str, Any], config: Optional[Dict[str, Any]] = None, **kwargs):
        """同步流式调用 Agent"""
        return self.graph.stream(input, config, **kwargs)

    # ==================== 状态管理 ====================

    async def get_state(self, thread_id: str):
        """获取指定线程的状态"""
        config = {"configurable": {"thread_id": thread_id}}
        return await self.graph.aget_state(config)

    def get_state_sync(self, thread_id: str):
        """同步获取指定线程的状态"""
        config = {"configurable": {"thread_id": thread_id}}
        return self.graph.get_state(config)

    async def update_state(self, thread_id: str, values: Dict[str, Any], as_node: Optional[str] = None):
        """更新指定线程的状态"""
        config = {"configurable": {"thread_id": thread_id}}
        return await self.graph.aupdate_state(config, values, as_node=as_node)

    async def get_state_history(self, thread_id: str, limit: int = 10):
        """获取状态历史"""
        config = {"configurable": {"thread_id": thread_id}}
        history = []
        async for state in self.graph.aget_state_history(config, limit=limit):
            history.append(state)
        return history


class ToolAgent(BaseAgent[TState]):
    """
    工具代理基类
    
    提供标准的工具绑定和执行模式
    """

    def __init__(
        self,
        llm: Optional[BaseChatModel] = None,
        tools: Optional[List[BaseTool]] = None,
        checkpointer: Optional[BaseCheckpointSaver] = None,
        bind_tools: bool = True,
        **kwargs
    ):
        super().__init__(llm=llm, tools=tools, checkpointer=checkpointer, **kwargs)
        
        # 自动绑定工具到 LLM
        if bind_tools and self._tools:
            self._llm_with_tools = self.llm.bind_tools(self._tools)
        else:
            self._llm_with_tools = self.llm

    @property
    def llm_with_tools(self) -> BaseChatModel:
        """获取绑定了工具的 LLM"""
        return self._llm_with_tools

    def add_tool(self, tool: BaseTool) -> None:
        """动态添加工具"""
        self._tools.append(tool)
        self._llm_with_tools = self.llm.bind_tools(self._tools)

    def remove_tool(self, tool_name: str) -> bool:
        """动态移除工具"""
        for i, tool in enumerate(self._tools):
            if tool.name == tool_name:
                self._tools.pop(i)
                self._llm_with_tools = self.llm.bind_tools(self._tools)
                return True
        return False
