"""消息过滤工具 - 清理传递给 Worker 的上下文。

根据 LangChain 最佳实践，移除 handoff/routing 消息可以：
- 减少 Worker 上下文窗口的噪音
- 避免 LLM 被路由逻辑干扰
- 节省 token 消耗
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage

# 需要过滤的路由节点名称
ROUTING_NODE_NAMES = frozenset([
    "intent_router",
    "guardrail",
    "supervisor",
])


def validate_worker_input(state: dict) -> tuple[str, str | None]:
    """Validate worker input state and extract the query.

    Args:
        state: The AgentState dict containing messages

    Returns:
        Tuple of (query_text, error_message).
        If error_message is not None, the worker should return an error response.
    """
    messages = state.get("messages")

    if not messages:
        return "", "No messages to process."

    if not isinstance(messages, list):
        return "", "Invalid messages format."

    last_message = messages[-1]
    if last_message is None:
        return "", "Last message is None."

    content = getattr(last_message, "content", None)
    if content is None:
        content = ""

    query = str(content).strip()
    if not query:
        return "", "Empty query received."

    return query, None


def make_error_response(error_msg: str) -> dict:
    """Create a standard error response for workers.

    Args:
        error_msg: The error message to include

    Returns:
        Dict with messages containing an AIMessage with the error
    """
    return {"messages": [AIMessage(content=error_msg)]}


def filter_messages_for_worker(
    messages: list[BaseMessage],
    keep_system: bool = True,
    keep_last_n_exchanges: int = 5,
) -> list[BaseMessage]:
    """过滤消息，为 Worker 创建干净的上下文。

    移除:
    - 路由节点的消息 (intent_router, guardrail, supervisor)
    - 空内容消息

    保留:
    - SystemMessage (可配置)
    - 最近 N 轮对话 (Human + AI 配对)

    Args:
        messages: 原始消息列表
        keep_system: 是否保留系统消息
        keep_last_n_exchanges: 保留最近几轮对话，0 表示不限制

    Returns:
        过滤后的消息列表
    """
    filtered: list[BaseMessage] = []

    for msg in messages:
        # 获取消息名称
        msg_name = getattr(msg, "name", "") or ""

        # 跳过路由节点消息
        if msg_name.lower() in ROUTING_NODE_NAMES:
            continue

        # 跳过空内容
        content = getattr(msg, "content", "")
        if not content or not str(content).strip():
            continue

        # 保留系统消息
        if isinstance(msg, SystemMessage):
            if keep_system:
                filtered.append(msg)
            continue

        # 保留其他消息
        filtered.append(msg)

    # 限制对话轮数
    if keep_last_n_exchanges > 0:
        # 分离系统消息和对话消息
        system_msgs = [m for m in filtered if isinstance(m, SystemMessage)]
        conv_msgs = [m for m in filtered if not isinstance(m, SystemMessage)]

        # 计算要保留的消息数 (每轮约 2 条: Human + AI)
        max_conv = keep_last_n_exchanges * 2
        if len(conv_msgs) > max_conv:
            conv_msgs = conv_msgs[-max_conv:]

        filtered = system_msgs + conv_msgs

    return filtered
