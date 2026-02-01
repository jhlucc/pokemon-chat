"""Test parallel worker execution via Send() API."""

from __future__ import annotations

from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage

from src.graph.nodes.rule_router import rule_route_parallel
from src.graph.nodes.supervisor import supervisor_node


class _NoLLM:
    """Mock LLM that should never be called for rule-routed queries."""

    def bind_tools(self, *args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("LLM should not be called for rule-routable parallel query")


def test_rule_route_parallel_detects_multiple_workers():
    """rule_route_parallel should return multiple workers when query matches multiple keyword groups."""
    # Query matches both stats (属性克制) and graph (进化)
    query = "皮卡丘的进化链和属性克制是什么？"
    allowed = ["rag_worker", "web_worker", "graph_worker", "stats_worker", "mcp_worker"]

    workers = rule_route_parallel(query, allowed)

    assert len(workers) >= 2, f"Expected multiple workers, got {workers}"
    assert "stats_worker" in workers, "Should detect stats keywords (属性克制)"
    assert "graph_worker" in workers, "Should detect graph keywords (进化)"


def test_rule_route_parallel_single_match_returns_single():
    """rule_route_parallel returns single worker when only one keyword group matches."""
    query = "皮卡丘的属性克制"
    allowed = ["rag_worker", "stats_worker", "graph_worker"]

    workers = rule_route_parallel(query, allowed)

    assert workers == ["stats_worker"], f"Expected single stats_worker, got {workers}"


def test_rule_route_parallel_no_match_returns_empty():
    """rule_route_parallel returns empty list when no keywords match."""
    query = "皮卡丘是什么？"
    allowed = ["rag_worker", "stats_worker", "graph_worker"]

    workers = rule_route_parallel(query, allowed)

    assert workers == [], f"Expected empty list, got {workers}"


def test_supervisor_returns_parallel_for_multi_keyword_query():
    """Supervisor should return __PARALLEL__ with parallel_workers for multi-keyword queries."""
    state = {
        "messages": [HumanMessage(content="皮卡丘的进化链和属性克制")],
        "next": "",
        "allowed_workers": ["web_worker", "rag_worker", "graph_worker", "stats_worker", "mcp_worker"],
    }

    with patch("src.graph.nodes.supervisor.build_chat_llm", return_value=_NoLLM()):
        result = supervisor_node(state)

    assert result["next"] == "__PARALLEL__", f"Expected __PARALLEL__, got {result['next']}"
    assert "parallel_workers" in result, "Should include parallel_workers list"
    assert len(result["parallel_workers"]) >= 2, f"Expected 2+ workers, got {result['parallel_workers']}"
    assert "stats_worker" in result["parallel_workers"]
    assert "graph_worker" in result["parallel_workers"]
    assert result.get("forward_directly") is True, "Should set forward_directly for parallel"


def test_supervisor_single_keyword_still_uses_single_route():
    """Supervisor should use single worker route when only one keyword matches."""
    state = {
        "messages": [HumanMessage(content="皮卡丘的属性克制")],
        "next": "",
        "allowed_workers": ["web_worker", "rag_worker", "graph_worker", "stats_worker"],
    }

    with patch("src.graph.nodes.supervisor.build_chat_llm", return_value=_NoLLM()):
        result = supervisor_node(state)

    assert result["next"] == "stats_worker", f"Expected stats_worker, got {result['next']}"
    assert result.get("forward_directly") is True
