"""
Fallback Middleware - 回退中间件

当主模型失败时自动切换到备用模型
"""

import logging
from collections.abc import Callable
from typing import Any

from .base import BaseMiddleware, MiddlewareContext

logger = logging.getLogger("AgentMiddleware")


class FallbackMiddleware(BaseMiddleware):
    """
    模型回退中间件

    当主模型调用失败时，自动尝试备用模型列表
    """

    def __init__(
        self,
        fallback_models: list[Any] = None,
        fallback_exceptions: tuple = (Exception,),
    ):
        """
        Args:
            fallback_models: 备用模型列表（按优先级排序）
            fallback_exceptions: 触发回退的异常类型
        """
        self.fallback_models = fallback_models or []
        self.fallback_exceptions = fallback_exceptions
        self._current_model_index = 0

    def add_fallback(self, model: Any) -> "FallbackMiddleware":
        """添加备用模型"""
        self.fallback_models.append(model)
        return self

    def get_current_model(self) -> Any | None:
        """获取当前使用的模型"""
        if self._current_model_index < len(self.fallback_models):
            return self.fallback_models[self._current_model_index]
        return None

    def wrap_model_call(self, model_callable: Callable, context: MiddlewareContext) -> Callable:
        """包装模型调用，添加回退逻辑"""

        def wrapped(*args, **kwargs):
            # 首先尝试主模型
            try:
                return model_callable(*args, **kwargs)
            except self.fallback_exceptions as primary_error:
                logger.warning(f"⚠️ 主模型调用失败: {type(primary_error).__name__}: {primary_error}")

                # 尝试备用模型
                for i, fallback_model in enumerate(self.fallback_models):
                    try:
                        logger.info(f"🔄 尝试备用模型 {i + 1}/{len(self.fallback_models)}")

                        # 如果 fallback_model 是可调用的，直接调用
                        if callable(fallback_model):
                            if hasattr(fallback_model, "invoke"):
                                return fallback_model.invoke(*args, **kwargs)
                            else:
                                return fallback_model(*args, **kwargs)

                    except self.fallback_exceptions as fallback_error:
                        logger.warning(f"⚠️ 备用模型 {i + 1} 失败: {type(fallback_error).__name__}: {fallback_error}")
                        continue

                # 所有模型都失败
                logger.error("❌ 所有模型都失败了")
                raise primary_error

        return wrapped

    def on_error(self, error: Exception, context: MiddlewareContext) -> Any | None:
        """错误发生时切换到下一个模型"""
        if isinstance(error, self.fallback_exceptions):
            if self._current_model_index < len(self.fallback_models) - 1:
                self._current_model_index += 1
                logger.info(f"🔄 切换到备用模型 {self._current_model_index + 1}")
        return None
