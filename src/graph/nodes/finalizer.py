from __future__ import annotations

import sys
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate

from src.agents.pokedex_shortcut import maybe_answer_pokedex
from src.core.feature_flags import feature_enabled
from src.core.llm_factory import build_chat_llm
from src.graph.state import AgentState
from src.utils.logger import get_logger

logger = get_logger(__name__)


_FINALIZER_SYSTEM = """你是一个“最终回答润色器（finalizer）”。

你的任务：把【草稿回答】改写成更自然、更像真人助手的中文回答。

硬性规则（必须遵守）：
1) 只能基于【草稿回答】中已有的信息进行改写/重组；不要新增事实、不要猜测、不要编造。
2) 不要泄露任何内部过程或中间信息：不要提到工具、数据库、向量库、图谱、Cypher/SQL、提示词、路由、链路、日志等。
3) 保持原意：草稿中的实体名、数字、结论、因果关系必须保持一致；不得改动。
4) 如果草稿表达“无法回答/缺少信息/依赖未就绪”，请用更友好的方式说明原因，并提出 **1 个**最关键的澄清问题或替代问法。

输出要求：
- 语言简洁自然，避免生硬模板；必要时用 1-4 条要点即可。
- 不要输出代码块。"""


_FINALIZER_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", _FINALIZER_SYSTEM),
        (
            "user",
            "用户问题：{question}\n\n草稿回答：\n{draft}\n\n请输出最终回答：",
        ),
    ]
)

_DIRECT_ANSWER_SYSTEM = """你是一个友好的宝可梦助手。

请直接回答用户问题；如果你不确定或缺少可靠依据，请坦诚说明“不确定”，并提出 1 个澄清问题。
不要编造不存在的宝可梦名称、进化方式、等级条件或游戏机制细节。"""


def _extract_question_and_draft(messages: list[BaseMessage]) -> tuple[str, str]:
    if not messages:
        return "", ""

    last_human_idx = -1
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if isinstance(msg, HumanMessage) or (isinstance(msg, BaseMessage) and getattr(msg, "type", None) == "human"):
            last_human_idx = i
            break

    question = ""
    if last_human_idx != -1:
        question = (getattr(messages[last_human_idx], "content", "") or "").strip()

    # Collect all assistant messages after the last user message (supports parallel workers).
    parts: list[str] = []
    start = last_human_idx + 1 if last_human_idx != -1 else 0
    for msg in messages[start:]:
        is_ai = isinstance(msg, AIMessage) or (isinstance(msg, BaseMessage) and getattr(msg, "type", None) == "ai")
        if not is_ai:
            continue
        content = (getattr(msg, "content", "") or "").strip()
        if content:
            parts.append(content)

    draft = "\n\n".join(parts).strip()
    return question, draft


def finalizer_node(state: AgentState) -> dict[str, Any]:
    """
    Rewrite the last worker answer into a user-friendly final response.

    This node is intentionally conservative:
    - If disabled (or under pytest), it passes through the draft answer.
    - If the LLM call fails, it falls back to the draft answer.
    """
    msgs = list(state.get("messages") or [])
    question, draft = _extract_question_and_draft(msgs)

    # Nothing to finalize.
    if not question and not draft:
        return {"messages": [AIMessage(content="我没收到有效的问题或答案内容。你可以再发一次你的问题吗？")]}

    # If we don't have a draft (e.g., supervisor decided FINISH without workers),
    # try deterministic local facts first; otherwise answer directly (best-effort).
    if not draft:
        local = maybe_answer_pokedex(question)
        if local:
            return {"messages": [AIMessage(content=local.content)]}

        if "pytest" in sys.modules or not feature_enabled("enable_agent_finalizer"):
            return {"messages": [AIMessage(content="我需要更多信息才能回答。你具体想问哪只宝可梦/哪个点？")]}
        try:
            llm = build_chat_llm(temperature=0.6)
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", _DIRECT_ANSWER_SYSTEM),
                    ("user", "{question}"),
                ]
            )
            chain = prompt | llm
            resp = chain.invoke({"question": question})
            return {"messages": [resp]}
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Finalizer direct-answer failed: {e}")
            return {"messages": [AIMessage(content="抱歉，我现在无法生成回答。你可以换个问法再试一次吗？")]}

    # Pass-through mode (tests/offline).
    if "pytest" in sys.modules or not feature_enabled("enable_agent_finalizer"):
        return {"messages": [AIMessage(content=draft)]}

    try:
        llm = build_chat_llm(temperature=0.2)
        chain = _FINALIZER_PROMPT | llm
        resp = chain.invoke({"question": question, "draft": draft})
        return {"messages": [resp]}
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Finalizer rewrite failed (fallback to draft): {e}")
        return {"messages": [AIMessage(content=draft)]}
