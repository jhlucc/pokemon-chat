from __future__ import annotations

from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage

from src.agents.chat_agent import PokemonKGChatAgent


class _FakeLLM:
    def __init__(self, *args, **kwargs):  # noqa: D401, ANN001
        pass

    def invoke(self, _input, **_kwargs):  # noqa: ANN001
        return AIMessage(content="ok")

    def with_structured_output(self, _schema):  # noqa: ANN001
        return self

    async def ainvoke(self, _input, **_kwargs):  # noqa: ANN001
        return {"status": "pass", "reason": None}


class _StubGraph:
    def __init__(self):
        self.last_config = None

    async def astream(self, _input, config, stream_mode=None):  # noqa: ANN001
        self.last_config = config
        yield {"messages": [AIMessage(content="ok")]}


@pytest.mark.asyncio
async def test_chat_agent_query_sets_recursion_limit_in_config():
    with (
        patch("src.agents.chat_agent.build_chat_llm", return_value=_FakeLLM()),
        patch("src.agents.chat_agent.PokemonLightRAG"),
        patch("src.agents.chat_agent.LiteBaseSearcher"),
        patch("src.agents.chat_agent.PokemonStatsAgent"),
        patch("src.agents.chat_agent.PokedexAgent"),
        patch("src.agents.chat_agent.TrainerAgent"),
        patch("src.agents.chat_agent.DeepAgent"),
    ):
        agent = PokemonKGChatAgent()
        stub = _StubGraph()
        agent._graph = stub  # type: ignore[assignment]

        chunks = []
        async for part in agent.query("hi", meta={"thread_id": "t", "user_id": "u"}):
            chunks.append(part)

        assert chunks == ["ok"]
        assert stub.last_config is not None
        assert stub.last_config.get("recursion_limit") == 25

