from __future__ import annotations

import re

_YEAR_RE = re.compile(r"20\\d{2}")

_TIME_SENSITIVE_KEYWORDS = (
    "最新",
    "最近",
    "现在",
    "当前",
    "今日",
    "今天",
    "本周",
    "本月",
    "活动",
    "公告",
    "新闻",
    "更新",
    "版本",
    "补丁",
    "环境",
    "meta",
    "赛季",
)


def should_web_search(text: str) -> bool:
    """
    Return True iff the query is likely time-sensitive.

    Keep it simple/deterministic: keyword + year detection.
    """
    raw = (text or "").strip()
    if not raw:
        return False

    low = raw.lower()
    if any(k in raw for k in _TIME_SENSITIVE_KEYWORDS):
        return True
    if _YEAR_RE.search(low):
        return True
    return False
