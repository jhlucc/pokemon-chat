from __future__ import annotations

from unittest.mock import patch

from langchain_core.messages import HumanMessage

from src.graph.nodes.finalizer import finalizer_node


def test_finalizer_uses_local_pokedex_when_no_draft_and_does_not_call_llm():
    # If the supervisor ever routes to FINISH without running a worker,
    # finalizer must NOT hallucinate facts via an LLM call.
    state = {"messages": [HumanMessage(content="喷火龙进化是什么")], "next": "FINISH"}

    with patch("src.graph.nodes.finalizer.build_chat_llm", side_effect=AssertionError("LLM should not be called")):
        out = finalizer_node(state)

    msg = out["messages"][-1]
    content = getattr(msg, "content", "")

    assert "小火龙" in content
    assert "火恐龙" in content
    assert "喷火龙" in content

    # Ensure we didn't hallucinate non-existent evolutions.
    assert "焰尾" not in content
    assert "烈焰龙" not in content
