from __future__ import annotations

import re
from collections.abc import Sequence


_YEAR_RE = re.compile(r"20\d{2}")

# Keywords grouped by worker - use tuples for immutability
_WEB_KEYWORDS = ("最新", "最近", "现在", "当前", "今日", "今天", "活动", "公告", "新闻", "更新", "版本", "补丁", "环境", "meta")
_STATS_KEYWORDS = ("属性", "克制", "弱点", "抗性", "相性", "倍率", "对战", "种族值", "伤害", "速度线")
_GRAPH_KEYWORDS = ("关系", "进化", "谁", "伙伴", "属于", "位于", "地区", "城镇")
_MCP_KEYWORDS = ("在哪", "哪里", "位置", "坐标", "地图", "现实", "真实")

# Mapping from keyword group to worker
_KEYWORD_WORKER_MAP = {
    "web": ("web_worker", _WEB_KEYWORDS),
    "stats": ("stats_worker", _STATS_KEYWORDS),
    "graph": ("graph_worker", _GRAPH_KEYWORDS),
    "mcp": ("mcp_worker", _MCP_KEYWORDS),
}


def _match_keyword(query: str, keywords: tuple[str, ...]) -> bool:
    """Check if any keyword appears in query.

    For CJK (Chinese/Japanese/Korean) keywords, direct substring matching works well
    because CJK languages don't use spaces between words.

    For ASCII keywords (like 'meta'), we check word boundaries to avoid
    false positives (e.g., 'metadata' should not match 'meta').
    """
    for kw in keywords:
        if kw in query:
            # ASCII keyword - check for word boundary
            if kw.isascii():
                # Build a pattern that matches the keyword at word boundaries
                pattern = rf"(?<![a-zA-Z]){re.escape(kw)}(?![a-zA-Z])"
                if re.search(pattern, query, re.IGNORECASE):
                    return True
            else:
                # CJK keyword - direct match is sufficient
                return True
    return False


def _detect_workers(query: str, allowed: set[str]) -> list[str]:
    """Detect all workers matching keywords in the query."""
    q = (query or "").strip()
    if not q:
        return []

    low = q.lower()
    matched = []

    # Check year pattern for web
    if _YEAR_RE.search(low) and "web_worker" in allowed:
        matched.append("web_worker")

    for _group, (worker, keywords) in _KEYWORD_WORKER_MAP.items():
        if worker in allowed and worker not in matched:
            if _match_keyword(q, keywords):
                matched.append(worker)

    return matched


def rule_route(query: str, allowed_workers: Sequence[str]) -> str | None:
    """
    Fast heuristic routing for the supervisor workflow.

    Returns a worker name if confident, else None (fallback to LLM router).
    """
    allowed = set([w for w in allowed_workers if isinstance(w, str)])
    matched = _detect_workers(query, allowed)

    # Single match: return the worker (backward compatible)
    if len(matched) == 1:
        return matched[0]

    # Multiple matches or no matches: let caller decide
    return None


def rule_route_parallel(query: str, allowed_workers: Sequence[str]) -> list[str]:
    """
    Detect all workers that should handle this query in parallel.

    Returns a list of worker names. Empty list means no confident match.
    Use this when parallel execution is enabled.
    """
    allowed = set([w for w in allowed_workers if isinstance(w, str)])
    return _detect_workers(query, allowed)
