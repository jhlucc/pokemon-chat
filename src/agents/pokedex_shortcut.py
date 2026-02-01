from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.agents.intent import Intent, classify_intent
from src.agents.pokemon_data import PokemonData, get_pokemon_data
from src.agents.pokemon_facts import evolution_chain, format_type_matchups


@dataclass(frozen=True, slots=True)
class PokedexAnswer:
    content: str
    used_entity: str | None = None
    used_sections: tuple[str, ...] = ()


_EVOLUTION_KWS = ("进化", "evolution", "evolve")
_TYPE_MATCHUP_KWS = ("属性相性", "相性", "弱点", "抗性", "克制")
_NAME_KWS = ("日文名", "英文名", "日语名", "英语名", "jp", "en")
_DETAIL_KWS = ("详细", "图鉴", "资料", "信息", "全部", "完整", "detail", "details", "full")

_TYPE_KWS = ("属性", "type")
_HW_KWS = ("身高", "体重", "height", "weight")
_ABILITY_KWS = ("特性", "隐藏特性", "ability")
_ID_KWS = ("编号", "图鉴编号", "全国图鉴", "#", "id")


def _wants_any(query: str, keywords: tuple[str, ...]) -> bool:
    q = (query or "").strip()
    if not q:
        return False
    low = q.lower()
    for kw in keywords:
        if kw in q:
            return True
        if kw.isascii() and kw.lower() in low:
            return True
    return False


def _resolve_record(name: str, *, data: PokemonData) -> dict | None:
    resolved = data.resolve_name(name) or name
    return data.get_by_cn_name(resolved)


def _is_noneish(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and value.strip().lower() in {"", "none", "null", "nan"}:
        return True
    return False


def _as_list(value: Any) -> list[str]:
    if _is_noneish(value):
        return []
    if isinstance(value, list):
        out: list[str] = []
        for v in value:
            if _is_noneish(v):
                continue
            s = str(v).strip()
            if s:
                out.append(s)
        return out
    if isinstance(value, str):
        s = value.strip()
        return [s] if s else []
    s = str(value).strip()
    return [s] if s else []


def _format_evolution_chat(rec: dict, *, data: PokemonData) -> str:
    name = (rec.get("chinese_name") or "").strip()
    if not name:
        return "我没找到这只宝可梦的进化信息。"

    chain = evolution_chain(name, data=data)
    if not chain:
        return f"我没找到{name}的进化信息。"
    if len(chain) == 1:
        return f"{name}不会再进化（最终形态）。"

    idx = chain.index(name) if name in chain else len(chain) - 1
    prev = chain[idx - 1] if idx > 0 else None
    nxt = chain[idx + 1] if idx < len(chain) - 1 else None

    if nxt:
        lead = f"{name}会进化成{nxt}。"
    else:
        lead = f"{name}不会再进化，是最终形态。"
        if prev:
            lead = f"{lead}它由{prev}进化而来。"

    return lead + "\n" + f"进化链：{' → '.join(chain)}"


def _format_detail_chat(rec: dict) -> str:
    name = (rec.get("chinese_name") or "").strip() or "未知宝可梦"
    pid = rec.get("id")
    pid_s = f"#{pid}" if isinstance(pid, int) else ""

    en = (rec.get("english_name") or "").strip()
    jp = (rec.get("japanese_name") or "").strip()
    types = "/".join(_as_list(rec.get("type"))) or "未知"

    h = rec.get("height")
    w = rec.get("weight")
    h_s = "" if _is_noneish(h) else str(h).strip()
    w_s = "" if _is_noneish(w) else str(w).strip()

    abilities = _as_list(rec.get("ability"))
    hidden = _as_list(rec.get("隐藏特性"))

    names: list[str] = []
    if pid_s:
        names.append(pid_s)
    if en:
        names.append(en)
    if jp:
        names.append(jp)
    names_s = f"（{', '.join(names)}）" if names else ""

    parts: list[str] = [f"{name}{names_s}的属性是{types}。"]
    if h_s or w_s:
        hw: list[str] = []
        if h_s:
            hw.append(f"身高{h_s} m")
        if w_s:
            hw.append(f"体重{w_s} kg")
        parts.append("，".join(hw) + "。")
    if abilities or hidden:
        if abilities:
            parts.append(f"特性：{'、'.join(abilities)}。")
        if hidden:
            parts.append(f"隐藏特性：{'、'.join(hidden)}。")

    return "".join(parts)


def maybe_answer_pokedex(query: str, *, data: PokemonData | None = None) -> PokedexAnswer | None:
    """
    Deterministic local Pokédex answer helper.

    This is deliberately conservative: it only triggers for Intent.POKEDEX_FACTS.
    Callers (workers) should decide when to prefer this over tool/LLM answers.
    """
    decision = classify_intent(query)
    if decision.intent != Intent.POKEDEX_FACTS:
        return None

    if decision.needs_clarification and decision.clarification_question:
        return PokedexAnswer(content=decision.clarification_question, used_sections=("clarify",))

    # Be conservative: avoid hijacking non-fact questions that merely mention an entity
    # (e.g. "皮卡丘和小智什么关系"). Those often belong to graph/rag, not pure facts.
    if decision.confidence < 0.8:
        return None

    if not decision.entities:
        return None

    data = data or get_pokemon_data()
    name = decision.entities[0]
    rec = _resolve_record(name, data=data)
    if not rec:
        return PokedexAnswer(
            content=f"我没找到宝可梦：{name}。你可以换个名字试试吗？", used_entity=name, used_sections=("not_found",)
        )

    wants_evo = _wants_any(query, _EVOLUTION_KWS)
    wants_type_matchups = _wants_any(query, _TYPE_MATCHUP_KWS)
    wants_names = _wants_any(query, _NAME_KWS)
    wants_detail = _wants_any(query, _DETAIL_KWS)

    wants_type = _wants_any(query, _TYPE_KWS)
    wants_hw = _wants_any(query, _HW_KWS)
    wants_ability = _wants_any(query, _ABILITY_KWS)
    wants_id = _wants_any(query, _ID_KWS)

    sections: list[str] = []
    used: list[str] = []

    # If user asks for "图鉴/详细", return a compact, human-readable summary.
    if wants_detail:
        sections.append(_format_detail_chat(rec))
        used.append("detail")

    # Evolution: keep it concise by default (avoid dumping the full pokedex card).
    if wants_evo:
        sections.append(_format_evolution_chat(rec, data=data))
        used.append("evolution")

    # Defensive matchup summary (best-effort from dataset).
    if wants_type_matchups:
        sections.append(format_type_matchups(rec))
        used.append("type_matchups")

    # Names: answer directly, do not dump unrelated fields.
    if wants_names and not wants_detail:
        cn = (rec.get("chinese_name") or "").strip() or name
        en = (rec.get("english_name") or "").strip()
        jp = (rec.get("japanese_name") or "").strip()
        parts: list[str] = []
        if en:
            parts.append(f"英文名 {en}")
        if jp:
            parts.append(f"日文名 {jp}")
        if parts:
            sections.append(f"{cn}的" + "，".join(parts) + "。")
            used.append("names")

    # Targeted basic questions.
    if (wants_type or wants_hw or wants_ability or wants_id) and not wants_detail:
        cn = (rec.get("chinese_name") or "").strip() or name
        if wants_id and isinstance(rec.get("id"), int):
            sections.append(f"{cn}的图鉴编号是 #{rec.get('id')}。")
            used.append("id")
        if wants_type:
            types = "/".join(_as_list(rec.get("type"))) or "未知"
            sections.append(f"{cn}的属性是{types}。")
            used.append("type")
        if wants_hw:
            h = rec.get("height")
            w = rec.get("weight")
            h_s = "" if _is_noneish(h) else str(h).strip()
            w_s = "" if _is_noneish(w) else str(w).strip()
            hw: list[str] = []
            if h_s:
                hw.append(f"身高{h_s} m")
            if w_s:
                hw.append(f"体重{w_s} kg")
            if hw:
                sections.append(f"{cn}的" + "，".join(hw) + "。")
                used.append("height_weight")
        if wants_ability:
            abilities = _as_list(rec.get("ability"))
            hidden = _as_list(rec.get("隐藏特性"))
            if abilities:
                sections.append(f"{cn}的特性是{'、'.join(abilities)}。")
                used.append("ability")
            if hidden:
                sections.append(f"{cn}的隐藏特性是{'、'.join(hidden)}。")
                used.append("hidden_ability")

    # If we didn't match any specific section (rare), give a short clarification.
    if not sections:
        cn = (rec.get("chinese_name") or "").strip() or name
        return PokedexAnswer(
            content=(f"你想了解{cn}的哪方面信息？比如：进化、属性、特性、身高体重、弱点/抗性。"),
            used_entity=cn,
            used_sections=("clarify",),
        )

    return PokedexAnswer(
        content="\n\n".join([s for s in sections if s.strip()]),
        used_entity=rec.get("chinese_name") or name,
        used_sections=tuple(used),
    )
