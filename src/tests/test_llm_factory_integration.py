from __future__ import annotations

from unittest.mock import patch

from langchain_core.messages import AIMessage
from langgraph.graph import END, START, MessagesState, StateGraph

from src.agents.base import BaseAgent
from src.agents.chat_agent import PokemonKGChatAgent


class _FakeLLM:
    def __init__(self, *args, **kwargs):  # noqa: D401, ANN001
        pass

    def invoke(self, _input, **_kwargs):  # noqa: ANN001
        return AIMessage(content="ok")


class _DummyAgent(BaseAgent[dict]):
    def _build_graph(self):  # noqa: D401
        g = StateGraph(MessagesState)
        g.add_node("noop", lambda s: s)
        g.add_edge(START, "noop")
        g.add_edge("noop", END)
        return g.compile(checkpointer=self.checkpointer)

    def get_info(self) -> dict:  # noqa: D401
        return {"name": "dummy"}


def test_baseagent_default_llm_uses_llm_factory():
    fake = _FakeLLM()
    with patch("src.agents.base.build_chat_llm", return_value=fake, create=True) as m:
        agent = _DummyAgent()
        assert agent.llm is fake
        m.assert_called_once()


def test_chat_agent_uses_llm_factory():
    fake = _FakeLLM()
    with (
        patch("src.agents.chat_agent.build_chat_llm", return_value=fake, create=True) as m,
        patch("src.agents.chat_agent.PokemonLightRAG"),
        patch("src.agents.chat_agent.LiteBaseSearcher"),
        patch("src.agents.chat_agent.PokemonStatsAgent"),
        patch("src.agents.chat_agent.PokedexAgent"),
        patch("src.agents.chat_agent.TrainerAgent"),
        patch("src.agents.chat_agent.DeepAgent"),
    ):
        PokemonKGChatAgent()
        m.assert_called()
