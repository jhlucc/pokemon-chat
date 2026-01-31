from __future__ import annotations

from functools import lru_cache
from typing import Any

from src.agents.pokemon_data import PokemonData, PokemonDetail, get_pokemon_data


_EVOLUTION_OVERRIDE_NEXT: dict[str, str] = {
    # The raw dataset has known gaps/inconsistencies for some baby Pokemon.
    # Keep this list small and targeted; prefer dataset truth otherwise.
    "皮丘": "皮卡丘",
}


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
        return [str(v).strip() for v in value if not _is_noneish(v) and str(v).strip()]
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    return [str(value).strip()] if str(value).strip() else []


def _as_float(value: Any) -> float | None:
    if _is_noneish(value):
        return None
    try:
        return float(value)
    except Exception:
        return None


@lru_cache(maxsize=1)
def _reverse_evolution_map() -> dict[str, list[str]]:
    """Map evolution target -> list of (previous stage) pokemon names."""
    data = get_pokemon_data()
    rev: dict[str, list[str]] = {}
    for rec in data.iter_all():
        nxt = rec.get("进化")
        if not isinstance(nxt, str) or _is_noneish(nxt):
            continue
        src = rec.get("chinese_name")
        if not isinstance(src, str) or _is_noneish(src):
            continue
        rev.setdefault(nxt.strip(), []).append(src.strip())

    # Apply small, explicit overrides (best-effort).
    for prev, nxt in _EVOLUTION_OVERRIDE_NEXT.items():
        rev.setdefault(nxt, []).append(prev)
    return rev


def evolution_chain(cn_name: str, *, data: PokemonData | None = None) -> list[str]:
    """Best-effort linear evolution chain around a Pokemon name."""
    data = data or get_pokemon_data()
    name = cn_name.strip()
    if not name:
        return []

    rev = _reverse_evolution_map()

    chain: list[str] = [name]
    seen: set[str] = {name}

    # Walk backwards (pre-evolutions). Prefer single-source chains only.
    cur = name
    while True:
        prevs = rev.get(cur) or []
        if len(prevs) != 1:
            break
        prev = prevs[0]
        if prev in seen:
            break
        chain.insert(0, prev)
        seen.add(prev)
        cur = prev

    # Walk forwards (evolutions).
    cur = name
    while True:
        rec = data.get_by_cn_name(cur)
        if not rec:
            break
        nxt = rec.get("进化")
        if isinstance(nxt, str) and not _is_noneish(nxt):
            nxt = nxt.strip()
        else:
            nxt = _EVOLUTION_OVERRIDE_NEXT.get(cur, "")
        if not nxt or nxt in seen:
            break
        chain.append(nxt)
        seen.add(nxt)
        cur = nxt

    return chain


def format_basic_facts(record: PokemonDetail) -> str:
    """Format a compact, deterministic pokedex-style summary."""
    name = (record.get("chinese_name") or "").strip() or "未知宝可梦"
    pid = record.get("id")

    types = _as_list(record.get("type"))
    types_str = "/".join(types) if types else "未知"

    height = _as_float(record.get("height"))
    weight = _as_float(record.get("weight"))

    abilities = _as_list(record.get("ability"))
    hidden = _as_list(record.get("隐藏特性"))

    title = f"## {name}"
    if isinstance(pid, int):
        title = f"## {name} (#{pid})"

    lines = [
        title,
        f"属性: {types_str}",
    ]
    if height is not None:
        lines.append(f"身高: {height:g} m")
    if weight is not None:
        lines.append(f"体重: {weight:g} kg")
    if abilities:
        lines.append(f"特性: {', '.join(abilities)}")
    if hidden:
        lines.append(f"隐藏特性: {', '.join(hidden)}")

    return "\n".join(lines)


def format_evolution(record: PokemonDetail, *, data: PokemonData | None = None) -> str:
    name = (record.get("chinese_name") or "").strip()
    if not name:
        return "进化链: 未知（数据缺少中文名字段）"

    chain = evolution_chain(name, data=data)
    if not chain:
        return f"进化链: 未找到 {name} 的进化信息"

    return "进化链: " + " → ".join(chain)


def format_type_matchups(record: PokemonDetail) -> str:
    """
    Format defensive type multipliers from the dataset (best-effort).

    Note: the raw dataset may contain inaccuracies; callers should treat this
    as informational only unless cross-validated.
    """
    raw = record.get("属性相性")
    if not isinstance(raw, dict) or not raw:
        return "属性相性: 暂无数据"

    # Sort by multiplier desc then type name for stable output.
    items: list[tuple[str, float]] = []
    for k, v in raw.items():
        if not isinstance(k, str) or _is_noneish(k):
            continue
        try:
            f = float(v)
        except Exception:
            continue
        items.append((k, f))
    if not items:
        return "属性相性: 暂无数据"

    items.sort(key=lambda kv: (-kv[1], kv[0]))
    lines = ["属性相性(被打倍率):"]
    for t, mult in items:
        lines.append(f"- {t}: {mult:g}x")
    return "\n".join(lines)
