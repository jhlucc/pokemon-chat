from __future__ import annotations

import sys
from unittest.mock import patch

from langchain_core.messages import AIMessage

from src.agents.chat_agent import PokemonKGChatAgent


class _FakeLLM:
    def __init__(self, *args, **kwargs):  # noqa: D401, ANN001
        pass

    def invoke(self, _input, **_kwargs):  # noqa: ANN001
        return AIMessage(content="ok")


def test_long_term_memory_middleware_not_imported_when_feature_off():
    # Ensure this is a meaningful test: if the agent imports the module, it will reappear.
    sys.modules.pop("src.agents.middleware.long_term_memory", None)

    with (
        patch("src.agents.chat_agent.build_chat_llm", return_value=_FakeLLM()),
        patch("src.agents.chat_agent.feature_enabled", return_value=False),
        patch("src.agents.chat_agent.PokemonLightRAG"),
        patch("src.agents.chat_agent.LiteBaseSearcher"),
        patch("src.agents.chat_agent.PokemonStatsAgent"),
        patch("src.agents.chat_agent.PokedexAgent"),
        patch("src.agents.chat_agent.TrainerAgent"),
        patch("src.agents.chat_agent.DeepAgent"),
    ):
        PokemonKGChatAgent()

    assert "src.agents.middleware.long_term_memory" not in sys.modules

