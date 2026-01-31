from __future__ import annotations

from unittest.mock import patch

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableLambda

from src.graph.nodes.stats_worker import stats_worker_node


def test_stats_worker_answers_type_matchups_deterministically():
    state = {"messages": [HumanMessage(content="电打水克制吗")]}

    # If the stats worker tries to call an LLM for a simple type matchup question,
    # this test should fail.
    no_llm = RunnableLambda(lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("LLM should not be used")))

    with patch("src.graph.nodes.stats_worker.build_chat_llm", return_value=no_llm):
        out = stats_worker_node(state)

    assert "效果拔群" in out["messages"][0].content
    assert "x" in out["messages"][0].content
