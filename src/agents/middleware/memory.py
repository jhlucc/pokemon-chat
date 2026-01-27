"""
Memory Middleware - 短期记忆管理中间件

实现消息修剪、摘要和删除策略
"""

import logging
from typing import Any

from langchain_core.messages import SystemMessage

from .base import BaseMiddleware, MiddlewareContext

logger = logging.getLogger("AgentMiddleware")


class MemoryMiddleware(BaseMiddleware):
    """
    短期记忆管理中间件

    策略:
    1. trim_messages: 前期处理 - 限制消息数量
    2. summarize: 中期处理 - 摘要旧消息
    3. delete_old: 后期处理 - 删除旧消息
    """

    def __init__(
        self,
        max_messages: int = 50,
        max_tokens: int = 4000,
        strategy: str = "trim",  # trim, summarize, delete
        keep_system_message: bool = True,
        summarizer_model: Any = None,
    ):
        """
        Args:
            max_messages: 最大消息数量
            max_tokens: 最大 token 数（估算）
            strategy: 策略 (trim/summarize/delete)
            keep_system_message: 是否保留系统消息
            summarizer_model: 用于摘要的模型
        """
        self.max_messages = max_messages
        self.max_tokens = max_tokens
        self.strategy = strategy
        self.keep_system_message = keep_system_message
        self.summarizer_model = summarizer_model

    def before_model(self, messages: list, context: MiddlewareContext) -> list:
        """在模型调用前处理消息"""
        if not messages:
            return messages

        if self.strategy == "trim":
            return self._trim_messages(messages)
        elif self.strategy == "summarize":
            return self._summarize_messages(messages, context)
        elif self.strategy == "delete":
            return self._delete_old_messages(messages)

        return messages

    def _trim_messages(self, messages: list) -> list:
        """修剪消息到最大数量"""
        if len(messages) <= self.max_messages:
            return messages

        # 保留系统消息
        system_messages = []
        other_messages = []

        for msg in messages:
            if self.keep_system_message and isinstance(msg, SystemMessage):
                system_messages.append(msg)
            else:
                other_messages.append(msg)

        # 保留最近的消息
        trimmed = other_messages[-(self.max_messages - len(system_messages)) :]

        logger.info(f"✂️ 消息修剪: {len(messages)} -> {len(system_messages) + len(trimmed)}")

        return system_messages + trimmed

    def _summarize_messages(self, messages: list, context: MiddlewareContext) -> list:
        """摘要旧消息"""
        if len(messages) <= self.max_messages:
            return messages

        if not self.summarizer_model:
            # 没有摘要模型，回退到修剪
            return self._trim_messages(messages)

        # 分离系统消息
        system_messages = []
        other_messages = []

        for msg in messages:
            if self.keep_system_message and isinstance(msg, SystemMessage):
                system_messages.append(msg)
            else:
                other_messages.append(msg)

        # 保留最近的消息，摘要旧消息
        keep_count = self.max_messages // 2
        messages_to_summarize = other_messages[:-keep_count]
        messages_to_keep = other_messages[-keep_count:]

        if messages_to_summarize:
            try:
                summary = self._create_summary(messages_to_summarize)
                summary_message = SystemMessage(content=f"[对话摘要]\n{summary}")

                logger.info(f"📝 消息摘要: {len(messages_to_summarize)} 条消息 -> 1 条摘要")

                return system_messages + [summary_message] + messages_to_keep
            except Exception as e:
                logger.warning(f"摘要生成失败: {e}")
                return self._trim_messages(messages)

        return system_messages + messages_to_keep

    def _create_summary(self, messages: list) -> str:
        """创建消息摘要"""
        # 构建摘要提示
        conversation = "\n".join([f"{type(m).__name__}: {m.content}" for m in messages])

        prompt = f"""请简洁地摘要以下对话的关键信息：

{conversation}

摘要："""

        response = self.summarizer_model.invoke(prompt)
        return response.content if hasattr(response, "content") else str(response)

    def _delete_old_messages(self, messages: list) -> list:
        """删除旧消息"""
        if len(messages) <= self.max_messages:
            return messages

        # 保留系统消息和最近的消息
        system_messages = []
        other_messages = []

        for msg in messages:
            if self.keep_system_message and isinstance(msg, SystemMessage):
                system_messages.append(msg)
            else:
                other_messages.append(msg)

        # 只保留最近的消息
        keep_count = self.max_messages - len(system_messages)
        kept = other_messages[-keep_count:]
        deleted_count = len(other_messages) - keep_count

        logger.info(f"🗑️ 删除旧消息: {deleted_count} 条")

        return system_messages + kept

    def after_agent(self, state: dict[str, Any], context: MiddlewareContext) -> dict[str, Any]:
        """Agent 执行后清理"""
        # 可在此实现额外的清理逻辑
        return state
