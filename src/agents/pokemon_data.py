from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

from src.core.settings import settings

PokemonDetail = dict[str, Any]


@dataclass(frozen=True, slots=True)
class PokemonData:
    """
    Local, deterministic Pokemon reference data.

    Backed by: `resources/data/raw_data/pokemon_detail.json` (dict keyed by CN name).
    """

    _by_cn_name: dict[str, PokemonDetail]
    _by_id: dict[int, PokemonDetail]

    @classmethod
    def load(cls, path: Path) -> "PokemonData":
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise TypeError("pokemon_detail.json must be a dict keyed by Chinese name")

        by_cn: dict[str, PokemonDetail] = {}
        by_id: dict[int, PokemonDetail] = {}

        for cn_name, rec in raw.items():
            if not isinstance(cn_name, str) or not isinstance(rec, dict):
                continue
            by_cn[cn_name] = rec

            pid = rec.get("id")
            try:
                pid_int = int(pid)
            except Exception:
                continue
            by_id[pid_int] = rec

        return cls(_by_cn_name=by_cn, _by_id=by_id)

    def get_by_cn_name(self, name: str) -> PokemonDetail | None:
        return self._by_cn_name.get(name)

    def get_by_id(self, pid: int | str) -> PokemonDetail | None:
        try:
            pid_int = int(pid)
        except Exception:
            return None
        return self._by_id.get(pid_int)

    def iter_all(self) -> Iterable[PokemonDetail]:
        return self._by_cn_name.values()


@lru_cache(maxsize=1)
def get_pokemon_data() -> PokemonData:
    """Load and cache the default Pokemon dataset for the current process."""
    path = settings.paths.raw_data / "pokemon_detail.json"
    return PokemonData.load(path)

