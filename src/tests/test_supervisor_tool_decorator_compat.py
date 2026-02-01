from __future__ import annotations

from src.graph.nodes.supervisor import _build_handoff_tools


def test_supervisor_handoff_tools_builder_is_compatible_with_langchain_core_tool_signature():
    tools = _build_handoff_tools(["rag_worker", "graph_worker"], support_parallel=True)
    names = {getattr(t, "name", "") for t in tools}

    assert "route_to_rag_worker" in names
    assert "route_to_graph_worker" in names
    assert "finish" in names
