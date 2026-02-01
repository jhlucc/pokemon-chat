from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

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
    _alias_to_cn_name: dict[str, str]

    @staticmethod
    def _norm(name: str) -> str:
        # Normalize for fuzzy-ish exact matching across CN/EN/JP.
        # Keep only alnum to remove spaces/punctuation; lowercase EN names.
        return "".join(ch.lower() for ch in name.strip() if ch.isalnum())

    @classmethod
    def load(cls, path: Path) -> PokemonData:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise TypeError("pokemon_detail.json must be a dict keyed by Chinese name")

        by_cn: dict[str, PokemonDetail] = {}
        by_id: dict[int, PokemonDetail] = {}
        alias: dict[str, str] = {}

        for cn_name, rec in raw.items():
            if not isinstance(cn_name, str) or not isinstance(rec, dict):
                continue
            by_cn[cn_name] = rec
            alias.setdefault(cls._norm(cn_name), cn_name)

            pid = rec.get("id")
            try:
                pid_int = int(pid)
            except Exception:
                continue
            by_id[pid_int] = rec

            # Build alias index (best-effort).
            for key in ("chinese_name", "english_name", "japanese_name"):
                val = rec.get(key)
                if isinstance(val, str) and val.strip():
                    alias.setdefault(cls._norm(val), cn_name)

        return cls(_by_cn_name=by_cn, _by_id=by_id, _alias_to_cn_name=alias)

    def get_by_cn_name(self, name: str) -> PokemonDetail | None:
        return self._by_cn_name.get(name)

    def get_by_id(self, pid: int | str) -> PokemonDetail | None:
        try:
            pid_int = int(pid)
        except Exception:
            return None
        return self._by_id.get(pid_int)

    def resolve_name(self, name: str) -> str | None:
        """
        Resolve CN/EN/JP names (or loose punctuation variants) to canonical CN name.

        Returns None if unknown.
        """
        if not isinstance(name, str) or not name.strip():
            return None
        # Fast-path: already a canonical CN key.
        if name in self._by_cn_name:
            return name
        return self._alias_to_cn_name.get(self._norm(name))

    def iter_all(self) -> Iterable[PokemonDetail]:
        return self._by_cn_name.values()


@lru_cache(maxsize=1)
def get_pokemon_data() -> PokemonData:
    """Load and cache the default Pokemon dataset for the current process."""
    path = settings.paths.raw_data / "pokemon_detail.json"
    return PokemonData.load(path)
