from __future__ import annotations

from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END

from src.agents.chat_agent import PokemonKGChatAgent


class _FakeLLM:
    """
    Minimal stub for ChatOpenAI-like objects used by PokemonKGChatAgent in unit tests.

    We intentionally return an invalid supervisor route to ensure the agent
    falls back safely (this test is RED before the structured-output migration).
    """

    def __init__(self, *args, **kwargs):  # noqa: D401, ANN001
        pass

    def invoke(self, _input, **_kwargs):  # noqa: ANN001
        # Old code path: prompt | llm | JsonOutputParser()
        return AIMessage(content='{"next":"not_a_real_node"}')

    def with_structured_output(self, _schema):  # noqa: ANN001
        # New code path (after refactor): prompt | llm.with_structured_output(...)
        # LangChain expects this to be a Runnable/callable; we deliberately raise to
        # exercise the supervisor's safe fallback behavior.
        def _raise(_input):  # noqa: ANN001
            raise ValueError("invalid structured output")

        return _raise


def test_supervisor_invalid_next_falls_back_to_finish():
    # Patch heavyweight subcomponents so we only test routing logic.
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

        state = {"messages": [HumanMessage(content="route me")]}
        result = agent._supervisor(state)

        # After migrating to structured outputs (Enum-constrained), invalid next values
        # must not leak into graph routing.
        assert result["next"] == END
