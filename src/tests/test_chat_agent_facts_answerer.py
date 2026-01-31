from __future__ import annotations

from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage

from src.agents.chat_agent import PokemonKGChatAgent


class _FakeLLM:
    def __init__(self, *args, **kwargs):  # noqa: D401, ANN001
        pass

    def invoke(self, _input, **_kwargs):  # noqa: ANN001
        return AIMessage(content="ok")


def test_facts_answerer_returns_height_and_weight_from_dataset():
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
        state = {"messages": [HumanMessage(content="皮卡丘 身高体重")]}
        out = agent._facts_answerer_node(state)  # type: ignore[attr-defined]
        msg = out["messages"][0]
        assert isinstance(msg, AIMessage)

        text = msg.content
        assert "0.4m" in text
        assert "6.0kg" in text
