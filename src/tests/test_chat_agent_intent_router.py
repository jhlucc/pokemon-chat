from __future__ import annotations

from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

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


@pytest.mark.asyncio
async def test_intent_router_routes_pokedex_to_facts_answerer():
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
        state = {"messages": [HumanMessage(content="皮卡丘 属性")]}
        out = await agent._intent_router_node(state)  # type: ignore[attr-defined]
        assert out["next"] == "facts_answerer"


@pytest.mark.asyncio
async def test_intent_router_routes_greeting_to_chat():
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
        state = {"messages": [HumanMessage(content="你好")]}
        out = await agent._intent_router_node(state)  # type: ignore[attr-defined]
        assert out["next"] == "chat"
