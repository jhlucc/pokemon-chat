from __future__ import annotations

import pytest

from src.agents.supervisor_agent import SupervisorAgent


@pytest.mark.asyncio
async def test_supervisor_agent_yields_deterministic_worker_output_when_no_llm_tokens():
    """
    The supervisor graph can route to deterministic workers (no LLM calls).
    In streaming mode, we still need to emit the final answer text.
    """
    agent = SupervisorAgent()
    meta = {"agent_constraints": {"allowed_workers": ["graph_worker"]}, "thread_id": "t"}

    chunks: list[str] = []
    async for part in agent.query("喷火龙进化是什么", meta=meta, history=[]):
        if isinstance(part, str):
            chunks.append(part)

    text = "".join(chunks)
    assert "小火龙" in text
    assert "火恐龙" in text
    assert "喷火龙" in text
    assert "MATCH" not in text
