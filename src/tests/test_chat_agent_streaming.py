from __future__ import annotations

from dataclasses import dataclass

import pytest

from src.agents.chat_agent import PokemonKGChatAgent


class _NoopMiddleware:
    def run_before_agent(self, input_message, _context):  # noqa: ANN001
        return input_message

    def run_after_agent(self, _input_message, _context):  # noqa: ANN001
        return None


@dataclass
class _Chunk:
    content: str


class _FakeGraph:
    async def astream(self, *_args, **_kwargs):  # noqa: ANN001
        raise AssertionError("query() should use astream_events for token streaming")
        if False:  # pragma: no cover
            yield None

    async def astream_events(self, *_args, **_kwargs):  # noqa: ANN001
        yield {"event": "on_chat_model_stream", "data": {"chunk": _Chunk(content="hel")}}
        yield {"event": "on_chat_model_stream", "data": {"chunk": _Chunk(content="lo")}}


@pytest.mark.asyncio
async def test_chat_agent_query_streams_tokens_via_astream_events():
    # Bypass expensive init; we only test the query() streaming behavior.
    agent = PokemonKGChatAgent.__new__(PokemonKGChatAgent)
    agent._graph = _FakeGraph()
    agent.middleware = _NoopMiddleware()

    parts: list[str] = []
    async for part in agent.query("hello", meta={"thread_id": "t", "user_id": "u"}):
        assert isinstance(part, str)
        parts.append(part)

    assert parts == ["hel", "lo"]
