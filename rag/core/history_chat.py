from typing import List

from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.memory import ChatMessageHistory
from langchain_core.messages import BaseMessage

from rag.core.prompts import get_system_prompt
from src.utils import logger

_log = logger.LogManager()


class HistoryManager:
    """
    使用 LangChain 的 ChatMessageHistory 来管理对话历史记录，并与
    SystemMessage、HumanMessage、AIMessage 进行集成。
    """

    def __init__(self, system_prompt=None):
        """
        初始化对话历史，并在第一条消息中插入System提示。
        :param system_prompt:自定义的system提示语，如果不提供则使用默认get_system_prompt。
        """
        self.history = ChatMessageHistory()
        self.system_prompt = system_prompt or get_system_prompt()
        # 将系统提示作为第一条消息
        self.history.add_message(SystemMessage(content=self.system_prompt))

    def add_user(self, content: str):
        """
        添加用户消息(HumanMessage)到对话历史。
        :param content: 用户输入的文本内容
        """
        self.history.add_message(HumanMessage(content=content))

    def add_ai(self, content: str):
        """
        添加AI消息 (AIMessage) 到对话历史。
        :param content: AI 返回的文本内容
        """
        self.history.add_message(AIMessage(content=content))

    def update_ai(self, content: str) -> List[BaseMessage]:
        """
        更新对话历史中最近一条AI消息的内容。如果最近一条消息不是AI消息，则添加新AI消息。
        :param content: 更新后的 AI 文本内容
        """
        if self.history.messages and isinstance(self.history.messages[-1], AIMessage):
            self.history.messages[-1] = AIMessage(content=content)
        else:
            self.add_ai(content)
        return self.history.messages

    def get_history_with_msg(self, msg: str, role: str = "user", max_rounds: int = None):
        """
        获取对话历史的副本并附加一条新的消息 (不会改变原始对话历史)。

        :param msg: 新消息内容
        :param role: 新消息的角色 (可选: "user", "assistant", "system")
        :param max_rounds: 控制最多包含多少轮的历史，每轮包含一条用户消息 + 一条 AI 消息
        :return: 返回一份不影响原记录的消息列表副本
        """
        if max_rounds is not None:
            # 每轮包含：用户消息 + AI 消息
            # 如果max_rounds=1，则最多包含最后2条消息；max_rounds=2，则是最后4条，以此类推
            relevant_messages = self.history.messages[-2 * max_rounds:]
        else:
            relevant_messages = list(self.history.messages)

        # 根据role创建不同类型的消息对象
        if role == "user":
            new_message = HumanMessage(content=msg)
        elif role == "assistant":
            new_message = AIMessage(content=msg)
        else:
            new_message = SystemMessage(content=msg)
        return relevant_messages + [new_message]

    def __str__(self) -> str:
        """
        将当前的消息记录转换为可读字符串。每条消息各占一行。
        """
        lines = []
        for message in self.history.messages:
            if isinstance(message, HumanMessage):
                role = "user"
            elif isinstance(message, AIMessage):
                role = "assistant"
            else:
                role = "unknown"
            # 将换行替换为空格
            content_single_line = message.content.replace('\n', ' ')
            lines.append(f"{role}: {content_single_line}")
        return "\n".join(lines)


if __name__ == "__main__":
    # 初始化 HistoryManager，并自动插入系统提示
    hm = HistoryManager()
    # 添加用户消息
    hm.add_user("Hello, how are you?")
    # 添加AI消息
    hm.add_ai("I'm just a bot, but I'm here to help. What can I do for you?")
    # 更新最后一条 AI 消息的内容
    hm.update_ai("I'm here to assist you with anything I can. How may I help you today?")
    # 获取一份带有额外用户消息的历史记录副本（但不修改原始记录）
    new_history = hm.get_history_with_msg("Can you tell me a joke?", role="user", max_rounds=1)
    print("=== 仅包含最后 1 轮（2 条消息）+ 新消息的副本内容 ===")
    for idx, msg in enumerate(new_history, start=1):
        print(f"{idx}. {msg.content}")
    # 打印完整的对话历史（已包含更新过的AI消息）
    print("\n=== 完整对话历史 ===")
    print(hm)
