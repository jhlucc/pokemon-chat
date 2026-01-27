"""
Agentic (long-term) memory.

This module is intentionally lightweight and offline-safe:
- Preferences are stored locally in SQLite (no external service required).
- Preference extraction is best-effort (can be extended to use an LLM).

The API is used by:
- `server/routers/chat_router.py` (add_conversation_turn, get_system_prompt_injection)
- unit tests in `src/tests/test_phase11.py`
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from pydantic import BaseModel, Field

from src.core.settings import settings
from src.utils.logger import get_logger

log = get_logger(__name__)


class UserPreferences(BaseModel):
    """
    Minimal preference schema expected by tests + UI.
    Extend as needed; keep defaults stable.
    """

    favorite_pokemon: list[str] = Field(default_factory=list)
    favorite_types: list[str] = Field(default_factory=list)
    response_style: str = "balanced"  # "brief" | "balanced" | "detailed"
    interests: list[str] = Field(default_factory=list)
    notes: str = ""


def _default_db_path() -> Path:
    # Prefer the "save" directory for writable persistence (works in Docker where
    # `/app/resources/save` is typically volume-mounted). Fall back to data_dir.
    base = getattr(settings.paths, "save_yaml_path", None) or getattr(settings.paths, "data_dir", None)
    return Path(base) / "memory" / "agentic_memory.sqlite"


class AgenticMemory:
    """
    SQLite-backed preference store.
    """

    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = Path(db_path) if db_path else _default_db_path()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.db_path))

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS user_preferences (
                    user_id TEXT PRIMARY KEY,
                    prefs_json TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS conversation_turns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    ts INTEGER NOT NULL
                )
                """
            )
            conn.commit()

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------

    def add_conversation_turn(self, user_id: str, role: str, content: str) -> None:
        """
        Store raw conversation turns (optional; useful for future extraction).
        """
        try:
            import time

            with self._connect() as conn:
                conn.execute(
                    "INSERT INTO conversation_turns(user_id, role, content, ts) VALUES(?,?,?,?)",
                    (user_id, role, content, int(time.time())),
                )
                conn.commit()
        except Exception as e:
            log.warning(f"add_conversation_turn failed (ignored): {e}")

    def get_preferences(self, user_id: str) -> UserPreferences:
        try:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT prefs_json FROM user_preferences WHERE user_id = ?",
                    (user_id,),
                ).fetchone()
            if not row:
                return UserPreferences()
            return UserPreferences.model_validate_json(row[0])
        except Exception as e:
            log.warning(f"get_preferences failed (ignored): {e}")
            return UserPreferences()

    def set_preferences(self, user_id: str, prefs: UserPreferences) -> None:
        try:
            with self._connect() as conn:
                conn.execute(
                    "INSERT INTO user_preferences(user_id, prefs_json) VALUES(?, ?) "
                    "ON CONFLICT(user_id) DO UPDATE SET prefs_json=excluded.prefs_json",
                    (user_id, prefs.model_dump_json()),
                )
                conn.commit()
        except Exception as e:
            log.warning(f"set_preferences failed (ignored): {e}")

    def extract_and_update_preferences(self, user_id: str) -> None:
        """
        Best-effort placeholder. In a full implementation, this can:
        - summarize recent turns
        - call an LLM to extract structured preferences
        """
        return

    def get_system_prompt_injection(self, user_id: str) -> str:
        prefs = self.get_preferences(user_id)
        # Keep the format stable; tests depend on this header.
        lines = ["[User Preferences]"]
        if prefs.response_style:
            lines.append(f"- response_style: {prefs.response_style}")
        if prefs.favorite_pokemon:
            lines.append(f"- favorite_pokemon: {', '.join(prefs.favorite_pokemon)}")
        if prefs.favorite_types:
            lines.append(f"- favorite_types: {', '.join(prefs.favorite_types)}")
        if prefs.interests:
            lines.append(f"- interests: {', '.join(prefs.interests)}")
        if prefs.notes:
            lines.append(f"- notes: {prefs.notes}")

        # If nothing meaningful, don't inject.
        if len(lines) <= 1:
            return ""
        return "\n" + "\n".join(lines)


# Global singleton used by chat_router.
_memory: AgenticMemory | None = None


def get_agentic_memory() -> AgenticMemory:
    global _memory
    if _memory is None:
        _memory = AgenticMemory()
    return _memory
