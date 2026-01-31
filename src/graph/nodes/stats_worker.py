import sys
from functools import lru_cache
from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate

from src.core.llm_factory import build_chat_llm
from src.graph.state import AgentState


@lru_cache
def _type_chart() -> dict[str, dict[str, float]]:
    # Lazy import to avoid pulling in heavy agent modules unless needed.
    from src.agents.pokemon_stats_agent import TYPE_EFFECTIVENESS

    return TYPE_EFFECTIVENESS


@lru_cache
def _known_types() -> tuple[str, ...]:
    chart = _type_chart()
    types: set[str] = set(chart.keys())
    for row in chart.values():
        types.update(row.keys())
    # Prefer longer tokens first to avoid substring collisions.
    return tuple(sorted(types, key=len, reverse=True))


def _extract_types_in_order(text: str) -> list[str]:
    q = (text or "").strip()
    if not q:
        return []

    hits: list[tuple[int, str]] = []
    for t in _known_types():
        idx = q.find(t)
        if idx != -1:
            hits.append((idx, t))

    hits.sort(key=lambda x: x[0])
    out: list[str] = []
    seen: set[str] = set()
    for _, t in hits:
        if t not in seen:
            out.append(t)
            seen.add(t)
    return out


def _type_multiplier(attack_type: str, defend_types: list[str]) -> float:
    chart = _type_chart()
    multiplier = 1.0
    row = chart.get(attack_type, {})
    for dt in defend_types:
        multiplier *= float(row.get(dt, 1.0))
    return multiplier


def _format_type_effectiveness(attack_type: str, defend_types: list[str]) -> str:
    multiplier = _type_multiplier(attack_type, defend_types)
    mult_str = f"{multiplier:g}"

    if multiplier == 0:
        effect = "无效"
    elif multiplier < 1:
        effect = "效果不好"
    elif multiplier > 1:
        effect = "效果拔群"
    else:
        effect = "普通"

    return f"{attack_type} -> {'/'.join(defend_types)}: {effect} ({mult_str}x)"


def _maybe_answer_type_matchup(query: str) -> str | None:
    q = (query or "").strip()
    if not q:
        return None

    # Fast heuristic: if the user is asking about type effectiveness and we can
    # extract at least two type tokens, answer deterministically.
    if not any(k in q for k in ("克制", "相性", "效果", "倍率", "打", "对")):
        return None

    attack: str | None = None
    defend: str | None = None

    if "打" in q:
        left, _, right = q.partition("打")
        left_types = _extract_types_in_order(left)
        right_types = _extract_types_in_order(right)
        attack = left_types[0] if left_types else None
        defend = right_types[0] if right_types else None
    elif "对" in q:
        left, _, right = q.partition("对")
        left_types = _extract_types_in_order(left)
        right_types = _extract_types_in_order(right)
        attack = left_types[0] if left_types else None
        defend = right_types[0] if right_types else None

    if not attack or not defend:
        types = _extract_types_in_order(q)
        if len(types) >= 2:
            attack, defend = types[0], types[1]

    if not attack or not defend:
        return None

    return _format_type_effectiveness(attack, [defend])


class StatsWorker:
    def __init__(self):
        self.llm = build_chat_llm(temperature=0.0)

    def analyze(self, query: str) -> str:
        # TODO: Implement structured data analysis (e.g. Pandas/SQL)
        return "Detailed statistical analysis is pending implementation."

    def __call__(self, state: AgentState) -> dict[str, Any]:
        messages = state["messages"]
        last_message = messages[-1]
        query = last_message.content

        deterministic = _maybe_answer_type_matchup(query)
        if deterministic:
            return {"messages": [AIMessage(content=deterministic)]}

        context = self.analyze(query)

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a data analyst. Provide insights based on data.\n\nAnalysis:\n{context}"),
                ("user", "{query}"),
            ]
        )

        chain = prompt | self.llm
        response = chain.invoke({"context": context, "query": query})

        return {"messages": [response]}


def stats_worker_node(state: AgentState):
    worker = get_stats_worker()
    return worker(state)


_stats_worker: StatsWorker | None = None


def get_stats_worker() -> StatsWorker:
    """
    Cached worker instance.

    NOTE: tests patch classes heavily; avoid caching under pytest to keep patches effective.
    """
    if "pytest" in sys.modules:
        return StatsWorker()
    global _stats_worker
    if _stats_worker is None:
        _stats_worker = StatsWorker()
    return _stats_worker


def clear_stats_worker_cache() -> None:
    global _stats_worker
    _stats_worker = None
