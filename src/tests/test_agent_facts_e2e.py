from __future__ import annotations

from unittest.mock import patch

import pytest

from src.agents.chat_agent import PokemonKGChatAgent


class _RaisingLLM:
    """LLM stub that hard-fails if anything tries to call it."""

    def invoke(self, *_args, **_kwargs):  # noqa: ANN001
        raise AssertionError("LLM should not be called for deterministic facts path")

    async def ainvoke(self, *_args, **_kwargs):  # noqa: ANN001
        raise AssertionError("LLM should not be called for deterministic facts path")

    def with_structured_output(self, _schema):  # noqa: ANN001
        return self


@pytest.mark.asyncio
async def test_agent_facts_path_is_offline_safe_no_llm():
    with (
        patch("src.agents.chat_agent.build_chat_llm", return_value=_RaisingLLM()),
        patch("src.agents.chat_agent.PokemonLightRAG"),
        patch("src.agents.chat_agent.LiteBaseSearcher"),
        patch("src.agents.chat_agent.PokemonStatsAgent"),
        patch("src.agents.chat_agent.PokedexAgent"),
        patch("src.agents.chat_agent.TrainerAgent"),
        patch("src.agents.chat_agent.DeepAgent"),
    ):
        agent = PokemonKGChatAgent()

        output = ""
        async for part in agent.query("皮卡丘 属性", meta={"thread_id": "t", "user_id": "u"}):
            if isinstance(part, str):
                output += part

        assert "电" in output
