"""
Tool Runtime Context

提供工具执行时的上下文信息，如 user_id, thread_id 等。
"""

from contextvars import ContextVar
from dataclasses import dataclass
from typing import Generic, TypeVar

T = TypeVar("T")


@dataclass
class ToolContext:
    """工具执行上下文"""

    user_id: str | None = None
    thread_id: str | None = None
    agent_name: str | None = None
    metadata: dict | None = None


# 使用 ContextVar 存储当前上下文
_current_tool_context: ContextVar[ToolContext | None] = ContextVar("current_tool_context", default=None)


def set_tool_context(context: ToolContext):
    """设置当前工具上下文"""
    _current_tool_context.set(context)


def get_tool_context() -> ToolContext | None:
    """获取当前工具上下文"""
    return _current_tool_context.get()


def clear_tool_context():
    """清除当前工具上下文"""
    _current_tool_context.set(None)


class ToolRuntime(Generic[T]):
    """
    工具运行时，用于在工具执行期间访问上下文。

    用法:
    ```python
    @tool
    def my_tool(runtime: ToolRuntime[MyContext]) -> str:
        user_id = runtime.context.user_id
        ...
    ```
    """

    def __init__(self, context: T):
        self._context = context

    @property
    def context(self) -> T:
        return self._context

    @classmethod
    def from_current(cls) -> "ToolRuntime[ToolContext]":
        """从当前上下文创建 ToolRuntime"""
        ctx = get_tool_context()
        if ctx is None:
            ctx = ToolContext()
        return cls(ctx)
