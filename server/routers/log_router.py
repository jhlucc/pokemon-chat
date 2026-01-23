from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from src.core.settings import settings

router = APIRouter(tags=["debug"])


_SECRET_PATTERNS: list[re.Pattern] = [
    # Common API key styles used by this project
    re.compile(r"sk-[A-Za-z0-9]{10,}"),
    re.compile(r"tvly-[A-Za-z0-9_-]{10,}"),
]


def _redact(text: str) -> str:
    for p in _SECRET_PATTERNS:
        text = p.sub("**REDACTED**", text)
    return text


def _find_latest_log_file(log_dir: Path) -> Optional[Path]:
    if not log_dir.exists():
        return None
    candidates = [p for p in log_dir.rglob("*.log") if p.is_file()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _tail_lines(path: Path, lines: int) -> str:
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read log file: {e}")

    parts = raw.splitlines()
    if lines > 0:
        parts = parts[-lines:]
    return "\n".join(parts)


@router.get("/log")
async def get_log(lines: int = Query(500, ge=50, le=5000)):
    """
    Return recent server logs for the frontend debug panel.
    NOTE: Best-effort redaction is applied.
    """
    log_dir = Path(settings.paths.log_dir)
    latest = _find_latest_log_file(log_dir)
    if latest is None:
        return {"log": ""}

    content = _tail_lines(latest, lines=lines)
    return {"log": _redact(content)}

