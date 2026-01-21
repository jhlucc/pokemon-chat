"""
Logging Middleware - 日志记录中间件

记录 Agent 和 Model 的执行过程
"""
import time
import logging
from typing import Any, Dict

from .base import BaseMiddleware, MiddlewareContext

logger = logging.getLogger("AgentMiddleware")


class LoggingMiddleware(BaseMiddleware):
    """
    日志记录中间件
    
    记录:
    - Agent 执行开始/结束
    - Model 调用开始/结束
    - 执行时间
    - 错误信息
    """
    
    def __init__(self, level: int = logging.INFO, log_messages: bool = False):
        """
        Args:
            level: 日志级别
            log_messages: 是否记录消息内容（可能包含敏感信息）
        """
        self.level = level
        self.log_messages = log_messages
        self._start_times: Dict[str, float] = {}
    
    def before_agent(self, state: Dict[str, Any], context: MiddlewareContext) -> Dict[str, Any]:
        key = f"agent_{context.thread_id}"
        self._start_times[key] = time.time()
        
        logger.log(
            self.level,
            f"🚀 Agent 开始执行 | thread={context.thread_id} | agent={context.agent_name}"
        )
        
        if self.log_messages and "messages" in state:
            msg_count = len(state.get("messages", []))
            logger.log(self.level, f"   消息数量: {msg_count}")
        
        return state
    
    def after_agent(self, state: Dict[str, Any], context: MiddlewareContext) -> Dict[str, Any]:
        key = f"agent_{context.thread_id}"
        elapsed = time.time() - self._start_times.pop(key, time.time())
        
        logger.log(
            self.level,
            f"✅ Agent 执行完成 | thread={context.thread_id} | 耗时={elapsed:.2f}s"
        )
        
        return state
    
    def before_model(self, messages: list, context: MiddlewareContext) -> list:
        key = f"model_{context.thread_id}"
        self._start_times[key] = time.time()
        
        logger.log(
            self.level,
            f"🤖 Model 调用开始 | thread={context.thread_id} | messages={len(messages)}"
        )
        
        return messages
    
    def after_model(self, response: Any, context: MiddlewareContext) -> Any:
        key = f"model_{context.thread_id}"
        elapsed = time.time() - self._start_times.pop(key, time.time())
        
        logger.log(
            self.level,
            f"💬 Model 调用完成 | thread={context.thread_id} | 耗时={elapsed:.2f}s"
        )
        
        return response
    
    def on_error(self, error: Exception, context: MiddlewareContext):
        logger.error(
            f"❌ 执行错误 | thread={context.thread_id} | error={type(error).__name__}: {error}"
        )
        return None  # 继续抛出错误
