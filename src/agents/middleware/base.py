"""
Base Middleware - 中间件基类

参考 LangGraph 1.x 中间件模式实现
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class MiddlewareContext:
    """中间件上下文"""

    agent_name: str = ""
    thread_id: str = ""
    user_id: str = ""
    extra: dict[str, Any] = None

    def __post_init__(self):
        if self.extra is None:
            self.extra = {}


class BaseMiddleware:
    """
    中间件基类

    生命周期钩子:
    - before_agent: Agent 执行前
    - after_agent: Agent 执行后
    - before_model: 模型调用前
    - after_model: 模型调用后
    - on_error: 发生错误时
    """

    def before_agent(self, state: dict[str, Any], context: MiddlewareContext) -> dict[str, Any]:
        """Agent 执行前处理"""
        return state

    def after_agent(self, state: dict[str, Any], context: MiddlewareContext) -> dict[str, Any]:
        """Agent 执行后处理"""
        return state

    def before_model(self, messages: list, context: MiddlewareContext) -> list:
        """模型调用前处理消息"""
        return messages

    def after_model(self, response: Any, context: MiddlewareContext) -> Any:
        """模型调用后处理响应"""
        return response

    def on_error(self, error: Exception, context: MiddlewareContext) -> Any | None:
        """
        错误处理

        Returns:
            None - 继续抛出错误
            Any - 返回替代值
        """
        return None

    def wrap_model_call(self, model_callable, context: MiddlewareContext):
        """
        包装模型调用

        可用于实现重试、回退等逻辑
        """

        def wrapped(*args, **kwargs):
            return model_callable(*args, **kwargs)

        return wrapped


class MiddlewareChain:
    """中间件链管理器"""

    def __init__(self, middlewares: list[BaseMiddleware] = None):
        self.middlewares = middlewares or []

    def add(self, middleware: BaseMiddleware) -> "MiddlewareChain":
        """添加中间件"""
        self.middlewares.append(middleware)
        return self

    def run_before_agent(self, state: dict[str, Any], context: MiddlewareContext) -> dict[str, Any]:
        """执行所有 before_agent 钩子"""
        for mw in self.middlewares:
            state = mw.before_agent(state, context)
        return state

    def run_after_agent(self, state: dict[str, Any], context: MiddlewareContext) -> dict[str, Any]:
        """执行所有 after_agent 钩子（逆序）"""
        for mw in reversed(self.middlewares):
            state = mw.after_agent(state, context)
        return state

    def run_before_model(self, messages: list, context: MiddlewareContext) -> list:
        """执行所有 before_model 钩子"""
        for mw in self.middlewares:
            messages = mw.before_model(messages, context)
        return messages

    def run_after_model(self, response: Any, context: MiddlewareContext) -> Any:
        """执行所有 after_model 钩子（逆序）"""
        for mw in reversed(self.middlewares):
            response = mw.after_model(response, context)
        return response

    def handle_error(self, error: Exception, context: MiddlewareContext) -> Any | None:
        """处理错误，返回第一个非 None 的结果"""
        for mw in self.middlewares:
            result = mw.on_error(error, context)
            if result is not None:
                return result
        return None

    def wrap_model_call(self, model_callable, context: MiddlewareContext):
        """
        包装模型调用，应用所有中间件的包装逻辑
        Middleware 顺序: [mw1, mw2]
        Wrap 顺序: mw1(mw2(model_callable))
        """
        wrapped = model_callable
        # 逆序应用，这样列表前面的中间件在最外层
        for mw in reversed(self.middlewares):
            wrapped = mw.wrap_model_call(wrapped, context)
        return wrapped
