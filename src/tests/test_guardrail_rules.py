from __future__ import annotations

from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.agents.chat_agent import PokemonKGChatAgent


class _NoLLMGuardrailChatOpenAI:
    def __init__(self, *args, **kwargs):  # noqa: D401, ANN001
        pass

    def invoke(self, _input, **_kwargs):  # noqa: ANN001
        return AIMessage(content="ok")

    def with_structured_output(self, _schema):  # noqa: ANN001
        raise AssertionError("Guardrail should not call LLM for this input")


@pytest.mark.asyncio
async def test_guardrail_blocks_obviously_offtopic_without_llm_call():
    with (
        patch("src.agents.chat_agent.build_chat_llm", return_value=_NoLLMGuardrailChatOpenAI()),
        patch("src.agents.chat_agent.PokemonLightRAG"),
        patch("src.agents.chat_agent.LiteBaseSearcher"),
        patch("src.agents.chat_agent.PokemonStatsAgent"),
        patch("src.agents.chat_agent.PokedexAgent"),
        patch("src.agents.chat_agent.TrainerAgent"),
        patch("src.agents.chat_agent.DeepAgent"),
    ):
        agent = PokemonKGChatAgent()
        out = await agent._guardrail_node({"messages": [HumanMessage(content="写个Python爬虫")]})
        assert out["next"] == "end_with_block"


@pytest.mark.asyncio
async def test_guardrail_allows_pokemon_entity_without_llm_call():
    with (
        patch("src.agents.chat_agent.build_chat_llm", return_value=_NoLLMGuardrailChatOpenAI()),
        patch("src.agents.chat_agent.PokemonLightRAG"),
        patch("src.agents.chat_agent.LiteBaseSearcher"),
        patch("src.agents.chat_agent.PokemonStatsAgent"),
        patch("src.agents.chat_agent.PokedexAgent"),
        patch("src.agents.chat_agent.TrainerAgent"),
        patch("src.agents.chat_agent.DeepAgent"),
    ):
        agent = PokemonKGChatAgent()
        out = await agent._guardrail_node({"messages": [HumanMessage(content="皮卡丘")]})
        assert out["next"] != "end_with_block"
