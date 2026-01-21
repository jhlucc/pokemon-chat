from typing import Literal
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig

from src.agents.base import BaseAgent
from src.agents.deep_agent.context import DeepContext
from src.core.settings import settings
from src.agents.tools.runtime import ToolRuntime
from src.agents.context.prompts import dynamic_prompt
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Nodes
async def research_node(state: DeepContext, config: RunnableConfig):
    """Execution step: Perform research or analysis"""
    logger.info(f"--- [DeepAgent] Researching: {state.get('topic')} (Iter: {state.get('iterations', 0)}) ---")
    
    # In a real impl, this would call an LLM with search tools
    # For now, we simulate a finding
    iteration = state.get("iterations", 0)
    return {
        "messages": [AIMessage(content=f"Research finding for iteration {iteration}")],
        "iterations": iteration + 1
    }

async def critique_node(state: DeepContext, config: RunnableConfig):
    """Critique step: Review findings"""
    logger.info("--- [DeepAgent] Critiquing ---")
    # Simulate critique
    return {
        "messages": [AIMessage(content="Critique: Needs more details on specific stats.")]
    }

def route_step(state: DeepContext) -> Literal["critique", "finalize"]:
    if state.get("iterations", 0) >= 2:
        return "finalize"
    return "critique"

async def finalize_node(state: DeepContext, config: RunnableConfig):
    """Finalize report"""
    logger.info("--- [DeepAgent] Finalizing ---")
    # Compile report
    return {
        "final_report": f"Final Report on {state.get('topic')}: ..."
    }

class DeepAgent(BaseAgent):
    """
    Deep Agent for in-depth analysis and research.
    Uses a Research -> Critique loop.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _build_graph(self):
        builder = StateGraph(DeepContext)
        
        builder.add_node("research", research_node)
        builder.add_node("critique", critique_node)
        builder.add_node("finalize", finalize_node)
        
        builder.add_edge(START, "research")
        
        builder.add_conditional_edges(
            "research",
            route_step,
            {
                "critique": "critique",
                "finalize": "finalize"
            }
        )
        
        builder.add_edge("critique", "research")
        builder.add_edge("finalize", END)
        
        return builder.compile()

    def get_info(self) -> dict:
        return {
            "name": "deep_agent",
            "description": "Performs deep research and analysis with self-correction."
        }

    async def query(self, question: str, **kwargs):
        # Initial state
        initial_state = {
            "topic": question,
            "messages": [HumanMessage(content=question)],
            "iterations": 0
        }
        config = {"configurable": {"thread_id": kwargs.get("thread_id", "default")}}
        
        async for output in self.graph.astream(initial_state, config):
            pass
            
        # Should return the final state or report
        final_state = await self.graph.aget_state(config)
        return final_state.values.get("final_report", "No report generated.")
