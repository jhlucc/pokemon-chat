import os
import unittest

if not os.getenv("RUN_INTEGRATION_TESTS"):
    raise unittest.SkipTest("Integration tests are skipped by default. Set RUN_INTEGRATION_TESTS=1 to run.")

from src.models.schemas import AgentResponse, Source


def test_agent_response():
    """Test creating and validating AgentResponse"""
    response = AgentResponse(
        content="Pikachu is an Electric-type Pokemon.",
        sources=[
            Source(title="Bulbapedia", url="https://bulbapedia.bulbagarden.net/wiki/Pikachu_(Pok%C3%A9mon)"),
            Source(title="Pokemon DB", score=0.9),
        ],
        confidence=0.95,
        metadata={"processed_by": "chat_agent"},
    )

    print("AgentResponse created successfully:")
    print(response.model_dump_json(indent=2))

    assert response.content == "Pikachu is an Electric-type Pokemon."
    assert len(response.sources) == 2
    assert response.sources[0].title == "Bulbapedia"
    assert response.confidence == 0.95


if __name__ == "__main__":
    test_agent_response()
