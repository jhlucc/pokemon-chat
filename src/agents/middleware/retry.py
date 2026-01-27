"""
Retry Middleware - 重试中间件

自动重试失败的模型调用
"""

import logging
import time
from collections.abc import Callable
from typing import Any

from .base import BaseMiddleware, MiddlewareContext

logger = logging.getLogger("AgentMiddleware")


class RetryMiddleware(BaseMiddleware):
    """
    重试中间件

    当模型调用失败时自动重试
    支持指数退避策略
    """

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        retryable_exceptions: tuple = (Exception,),
    ):
        """
        Args:
            max_retries: 最大重试次数
            base_delay: 基础延迟（秒）
            max_delay: 最大延迟（秒）
            exponential_base: 指数退避基数
            retryable_exceptions: 可重试的异常类型
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.retryable_exceptions = retryable_exceptions

    def _calculate_delay(self, attempt: int) -> float:
        """计算延迟时间（指数退避）"""
        delay = self.base_delay * (self.exponential_base**attempt)
        return min(delay, self.max_delay)

    def wrap_model_call(self, model_callable: Callable, context: MiddlewareContext) -> Callable:
        """包装模型调用，添加重试逻辑"""

        def wrapped(*args, **kwargs):
            last_exception = None

            for attempt in range(self.max_retries + 1):
                try:
                    return model_callable(*args, **kwargs)
                except self.retryable_exceptions as e:
                    last_exception = e

                    if attempt < self.max_retries:
                        delay = self._calculate_delay(attempt)
                        logger.warning(
                            f"🔄 模型调用失败，{delay:.1f}s 后重试 "
                            f"| attempt={attempt + 1}/{self.max_retries} "
                            f"| error={type(e).__name__}: {e}"
                        )
                        time.sleep(delay)
                    else:
                        logger.error(f"❌ 重试次数已用尽 | attempts={self.max_retries} | error={type(e).__name__}: {e}")

            raise last_exception

        return wrapped

    def on_error(self, error: Exception, context: MiddlewareContext) -> Any | None:
        """记录错误但不处理（让 wrap_model_call 处理重试）"""
        return None
