from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from src.agents.pokemon_entities import extract_pokemon_entities


class Intent(str, Enum):
    POKEDEX_FACTS = "pokedex_facts"
    TEAM_BUILDING = "team_building"
    WEB_SEARCH = "web_search"
    CHAT = "chat"
    UNKNOWN = "unknown"


class IntentDecision(BaseModel):
    intent: Intent
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    reason: str | None = None

    # Extracted/recognized entities (canonical CN names).
    entities: list[str] = Field(default_factory=list)

    # Filled by Task 7 when we can ask a targeted clarification question.
    needs_clarification: bool = False
    clarification_question: str | None = None


_WEB_KEYWORDS = ("最新", "最近", "现在", "活动", "新闻", "更新", "版本", "环境", "补丁", "公告")
_TEAM_KEYWORDS = ("队伍", "组队", "队", "配招", "阵容", "覆盖", "克制", "对战", "上分", "强度")
_FACT_KEYWORDS = ("属性", "身高", "体重", "特性", "隐藏特性", "进化", "图鉴", "编号", "弱点", "抗性", "相性")
_CHAT_KEYWORDS = ("你好", "您好", "hello", "hi", "hey", "早上好", "晚上好", "在吗")


def classify_intent(text: str, entities: list[str] | None = None) -> IntentDecision:
    """
    Cheap, deterministic intent classifier (rules-first).

    This is deliberately conservative: if uncertain, returns UNKNOWN so the graph
    can fall back to the supervisor/LLM.
    """
    raw = (text or "").strip()
    if not raw:
        return IntentDecision(intent=Intent.UNKNOWN, confidence=0.0, reason="empty")

    ents = list(entities) if entities is not None else extract_pokemon_entities(raw)
    low = raw.lower()

    if any(k in raw for k in _WEB_KEYWORDS):
        return IntentDecision(intent=Intent.WEB_SEARCH, confidence=0.9, reason="time_sensitive", entities=ents)

    if any(k in raw for k in _TEAM_KEYWORDS):
        return IntentDecision(intent=Intent.TEAM_BUILDING, confidence=0.8, reason="team_keywords", entities=ents)

    if any(k in low for k in _CHAT_KEYWORDS):
        return IntentDecision(intent=Intent.CHAT, confidence=0.7, reason="greeting", entities=ents)

    if ents and any(k in raw for k in _FACT_KEYWORDS):
        return IntentDecision(intent=Intent.POKEDEX_FACTS, confidence=0.85, reason="entity+fact_keywords", entities=ents)

    # If we at least recognized an entity, assume pokedex facts by default (conservative).
    if ents:
        return IntentDecision(intent=Intent.POKEDEX_FACTS, confidence=0.6, reason="entity_detected", entities=ents)

    return IntentDecision(intent=Intent.UNKNOWN, confidence=0.3, reason="no_match", entities=ents)

