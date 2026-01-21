from typing import TypedDict, List, Annotated
from langchain_core.messages import AnyMessage
import operator

class DeepContext(TypedDict):
    """Deep Agent State"""
    # 消息历史，追加模式
    messages: Annotated[List[AnyMessage], operator.add]
    # 当前研究主题
    topic: str
    # 最终报告
    final_report: str
    # 迭代次数
    iterations: int
    # 用户 ID
    user_id: str
    # 线程 ID
    thread_id: str
