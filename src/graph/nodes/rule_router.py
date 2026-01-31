from __future__ import annotations

import re
from collections.abc import Sequence


_YEAR_RE = re.compile(r"20\d{2}")

_WEB_KEYWORDS = ("最新", "最近", "现在", "当前", "今日", "今天", "活动", "公告", "新闻", "更新", "版本", "补丁", "环境", "meta")
_STATS_KEYWORDS = ("属性", "克制", "弱点", "抗性", "相性", "倍率", "对战", "种族值", "伤害", "速度")
_GRAPH_KEYWORDS = ("关系", "进化", "谁", "伙伴", "属于", "位于", "地区", "城镇")
_MCP_KEYWORDS = ("在哪", "哪里", "位置", "坐标", "地图", "现实", "真实")


def rule_route(query: str, allowed_workers: Sequence[str]) -> str | None:
    """
    Fast heuristic routing for the supervisor workflow.

    Returns a worker name if confident, else None (fallback to LLM router).
    """
    q = (query or "").strip()
    if not q:
        return None

    allowed = set([w for w in allowed_workers if isinstance(w, str)])
    low = q.lower()

    if any(k in q for k in _WEB_KEYWORDS) or _YEAR_RE.search(low):
        return "web_worker" if "web_worker" in allowed else None

    if any(k in q for k in _STATS_KEYWORDS):
        return "stats_worker" if "stats_worker" in allowed else None

    if any(k in q for k in _GRAPH_KEYWORDS):
        return "graph_worker" if "graph_worker" in allowed else None

    if any(k in q for k in _MCP_KEYWORDS):
        return "mcp_worker" if "mcp_worker" in allowed else None

    return None
