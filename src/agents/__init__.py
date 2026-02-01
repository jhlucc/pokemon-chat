"""
Agents package.

Keep package import cheap:
- Importing `src.agents.*` submodules should NOT eagerly register/initialize all agents.
- Access `agent_manager` via a lazy attribute for backward compatibility.
"""

from __future__ import annotations

from typing import Any

__all__ = ["agent_manager"]


def __getattr__(name: str) -> Any:
    if name == "agent_manager":
        # Lazy import: agent registration can pull in heavy deps (LangGraph/LLMs).
        from src.agents.manager import agent_manager as _agent_manager

        return _agent_manager
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
