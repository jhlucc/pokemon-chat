from __future__ import annotations

from langchain_core.messages import HumanMessage

import pytest

from src.agents.deep_agent.graph import DeepAgent


@pytest.mark.asyncio
async def test_deep_agent_uses_local_dataset_facts():
    agent = DeepAgent()

    # Keep it deterministic + fast.
    result = await agent.graph.ainvoke({"topic": "皮卡丘", "messages": [HumanMessage(content="皮卡丘")], "max_depth": 1})
    report = result.get("final_report") or ""

    # Dataset-backed ability that the old toy knowledge did not include.
    assert "静电" in report or "避雷针" in report
