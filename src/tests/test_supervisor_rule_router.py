from __future__ import annotations

from unittest.mock import patch

from langchain_core.messages import HumanMessage

from src.graph.nodes.supervisor import supervisor_node


class _NoLLM:
    def with_structured_output(self, _schema):  # noqa: ANN001
        raise AssertionError("LLM supervisor should not be used for this rule-routable query")


def test_supervisor_rules_route_web_queries():
    state = {
        "messages": [HumanMessage(content="宝可梦最新活动是什么？")],
        "next": "",
        "allowed_workers": ["web_worker", "rag_worker", "graph_worker", "stats_worker", "mcp_worker"],
    }
    with patch("src.graph.nodes.supervisor.build_chat_llm", return_value=_NoLLM()):
        out = supervisor_node(state)
    assert out["next"] == "web_worker"


def test_supervisor_rules_route_stats_queries():
    state = {
        "messages": [HumanMessage(content="皮卡丘 属性")],
        "next": "",
        "allowed_workers": ["web_worker", "rag_worker", "graph_worker", "stats_worker", "mcp_worker"],
    }
    with patch("src.graph.nodes.supervisor.build_chat_llm", return_value=_NoLLM()):
        out = supervisor_node(state)
    assert out["next"] == "stats_worker"

