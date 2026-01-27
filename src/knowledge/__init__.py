"""
src.knowledge

Keep this package import lightweight.

Importing `src.knowledge` happens implicitly whenever any submodule under
`src.knowledge.*` is imported. Historically, this module eagerly imported the
GraphRAG/LightRAG stack, which significantly slowed down server startup and
introduced optional dependency failures.

Prefer importing from the specific submodule you need, or rely on the lazy
exports provided here.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "PokemonLightRAG",
    "get_lightrag_instance",
]


_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "PokemonLightRAG": ("src.knowledge.graphrag", "PokemonLightRAG"),
    "get_lightrag_instance": ("src.knowledge.graphrag", "get_lightrag_instance"),
}


def __getattr__(name: str) -> Any:  # pragma: no cover
    target = _LAZY_ATTRS.get(name)
    if not target:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    mod_name, attr_name = target
    mod = import_module(mod_name)
    value = getattr(mod, attr_name)
    globals()[name] = value  # cache on first use
    return value
