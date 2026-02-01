from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any, Generic, TypeVar

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import MessagesState
from langgraph.graph.state import CompiledStateGraph

from src.core.llm_factory import build_chat_llm

# Type variable for state schema
TState = TypeVar("TState", bound=dict[str, Any])


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
        llm: BaseChatModel | None = None,
        tools: list[BaseTool] | None = None,
        checkpointer: BaseCheckpointSaver | None = None,
        **kwargs,
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
        self._graph: CompiledStateGraph | None = None

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
    def tools(self) -> list[BaseTool]:
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
        return build_chat_llm(temperature=0.7)

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

    async def ainvoke(self, input: dict[str, Any], config: dict[str, Any] | None = None) -> dict[str, Any]:
        """异步调用 Agent"""
        return await self.graph.ainvoke(input, config)

    def invoke(self, input: dict[str, Any], config: dict[str, Any] | None = None) -> dict[str, Any]:
        """同步调用 Agent"""
        return self.graph.invoke(input, config)

    async def astream(self, input: dict[str, Any], config: dict[str, Any] | None = None) -> AsyncIterator[Any]:
        """异步流式调用 Agent"""
        async for chunk in self.graph.astream(input, config):
            yield chunk

    def stream(self, input: dict[str, Any], config: dict[str, Any] | None = None, **kwargs):
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

    async def update_state(self, thread_id: str, values: dict[str, Any], as_node: str | None = None):
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

    async def query(
        self, query: str, meta: dict[str, Any] = None, history: list[dict[str, Any]] = None
    ) -> AsyncIterator[str]:
        """
        Unified query interface for API endpoints.
        Uses `astream_events` to yield tokens and internal steps (status).
        """
        from langchain_core.messages import HumanMessage

        input_state = {"messages": [HumanMessage(content=query)]}
        config = {"configurable": {"thread_id": meta.get("thread_id", "default")}}

        # astream_events (v2) allows seeing internal events
        async for event in self.graph.astream_events(input_state, config, version="v1"):
            kind = event["event"]

            if kind == "on_chat_model_stream":
                # Token streaming
                content = event["data"]["chunk"].content
                if content:
                    yield content

            elif kind == "on_tool_start":
                # Tool execution (Status update)
                tool_name = event["name"]
                tool_input = event["data"].get("input")
                # Yield a status update chunk
                yield {
                    "status": "tool_start",
                    "status_text": f"正在使用工具 {tool_name}...",
                    "tool": tool_name,
                    "input": tool_input,
                }
            elif kind == "on_tool_end":
                tool_name = event["name"]
                # Yield a status update chunk
                yield {"status": "tool_end", "status_text": f"工具 {tool_name} 执行完成", "tool": tool_name}


class ToolAgent(BaseAgent[TState]):
    """
    工具代理基类

    提供标准的工具绑定和执行模式
    """

    def __init__(
        self,
        llm: BaseChatModel | None = None,
        tools: list[BaseTool] | None = None,
        checkpointer: BaseCheckpointSaver | None = None,
        bind_tools: bool = True,
        **kwargs,
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

    # ============ 默认 tool-loop 图构建 ============

    def _build_graph(self) -> CompiledStateGraph:
        """默认的 tool-loop 图结构，子类可覆盖。

        图结构:
            START → agent → (has_tool_calls?) → run_tool → agent
                         ↘ (no tool calls) → END
        """
        workflow = StateGraph(MessagesState)
        workflow.add_node("agent", self._call_model)
        workflow.add_node("run_tool", self._run_tool)

        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges(
            "agent",
            self._should_continue,
            {"run_tool": "run_tool", "end": END},
        )
        workflow.add_edge("run_tool", "agent")

        return workflow.compile(checkpointer=self._checkpointer)

    def _call_model(self, state: dict) -> dict:
        """调用 LLM，子类可覆盖以定制行为"""
        messages = state["messages"]
        response = self._llm_with_tools.invoke(messages)
        return {"messages": [response]}

    def _should_continue(self, state: dict) -> str:
        """判断是否继续执行工具，子类可覆盖"""
        last_msg = state["messages"][-1]
        if not getattr(last_msg, "tool_calls", None):
            return "end"
        return "run_tool"

    def _run_tool(self, state: dict) -> dict:
        """执行工具调用，子类可覆盖"""
        new_messages = []
        tool_calls = state["messages"][-1].tool_calls
        tool_map = {t.name: t for t in self._tools}

        for call in tool_calls:
            tool = tool_map.get(call["name"])
            if tool:
                try:
                    result = tool.invoke(call["args"])
                    new_messages.append({
                        "role": "tool",
                        "name": call["name"],
                        "content": str(result),
                        "tool_call_id": call["id"],
                    })
                except Exception as e:
                    new_messages.append({
                        "role": "tool",
                        "name": call["name"],
                        "content": f"工具执行错误: {e}",
                        "tool_call_id": call["id"],
                    })

        return {"messages": new_messages}

    def get_info(self) -> dict[str, Any]:
        """返回 Agent 元信息，子类可覆盖"""
        return {
            "name": self.__class__.__name__,
            "description": self.__class__.__doc__ or "",
            "tools": [t.name for t in self._tools] if self._tools else [],
        }
