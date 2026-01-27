"""
Pokemon Deep Research Context

State definition for the Pokemon Deep Research agent.
"""

from typing import Annotated, Any

from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages
from typing_extensions import TypedDict


class DeepContext(TypedDict, total=False):
    """State for Pokemon Deep Research Agent"""

    # Core research fields
    topic: str
    messages: Annotated[list[BaseMessage], add_messages]
    iterations: int

    # Research parameters
    breadth: int  # Number of parallel queries
    depth: int  # Current depth level
    max_depth: int

    # Accumulated learnings
    learnings: list[str]
    research_directions: list[str]
    sources: list[str]

    # Pokemon-specific context
    pokemon_entities: list[str]  # Discovered Pokemon names
    type_analysis: dict[str, Any]
    battle_insights: list[str]

    # Final output
    final_report: str | None
