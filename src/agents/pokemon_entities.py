from __future__ import annotations

from functools import lru_cache

from src.agents.pokemon_data import PokemonData, get_pokemon_data


@lru_cache(maxsize=1)
def _alias_keys_by_len_desc() -> tuple[str, ...]:
    data = get_pokemon_data()
    # Prefer longer aliases to avoid substring collisions.
    return tuple(sorted(data._alias_to_cn_name.keys(), key=len, reverse=True))


def extract_pokemon_entities(text: str, *, max_entities: int = 5) -> list[str]:
    """
    Extract Pokemon entities from user text.

    Strategy:
    - Normalize text to an alnum-only string (lowercased) to match CN/EN/JP aliases.
    - Scan for known aliases; prefer longer matches.
    - Return canonical CN names, de-duped in first-appearance order.
    """
    if not isinstance(text, str) or not text.strip():
        return []

    data = get_pokemon_data()
    norm_text = PokemonData._norm(text)
    if not norm_text:
        return []

    matches: list[tuple[int, int, str]] = []
    for alias in _alias_keys_by_len_desc():
        start = norm_text.find(alias)
        while start != -1:
            end = start + len(alias)
            # Skip if overlaps with an already-accepted (longer) match.
            if any(not (end <= s or start >= e) for s, e, _ in matches):
                start = norm_text.find(alias, start + 1)
                continue
            matches.append((start, end, data._alias_to_cn_name[alias]))
            start = norm_text.find(alias, end)

        if len(matches) >= max_entities:
            break

    matches.sort(key=lambda t: t[0])
    out: list[str] = []
    seen: set[str] = set()
    for _s, _e, cn in matches:
        if cn in seen:
            continue
        out.append(cn)
        seen.add(cn)
        if len(out) >= max_entities:
            break
    return out

