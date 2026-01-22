"""
Pokemon Deep Research Context

State definition for the Pokemon Deep Research agent.
"""
from typing import List, Dict, Any, Optional, Annotated
from typing_extensions import TypedDict
from langgraph.graph import add_messages
from langchain_core.messages import BaseMessage


class DeepContext(TypedDict, total=False):
    """State for Pokemon Deep Research Agent"""
    # Core research fields
    topic: str
    messages: Annotated[List[BaseMessage], add_messages]
    iterations: int
    
    # Research parameters
    breadth: int  # Number of parallel queries
    depth: int    # Current depth level
    max_depth: int
    
    # Accumulated learnings
    learnings: List[str]
    research_directions: List[str]
    sources: List[str]
    
    # Pokemon-specific context
    pokemon_entities: List[str]  # Discovered Pokemon names
    type_analysis: Dict[str, Any]
    battle_insights: List[str]
    
    # Final output
    final_report: Optional[str]
