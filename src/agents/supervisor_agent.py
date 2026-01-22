from typing import Dict, Any
from langgraph.graph.state import CompiledStateGraph
from src.agents.base import BaseAgent
from src.graph.workflow import workflow
from src.core.settings import settings

class SupervisorAgent(BaseAgent):
    """
    Supervisor Agent that uses the modular LangGraph workflow.
    """
    def _build_graph(self) -> CompiledStateGraph:
        """
        Builds the graph using the pre-defined workflow and the agent's checkpointer.
        """
        return workflow.compile(checkpointer=self.checkpointer)

    def get_info(self) -> dict:
        return {
            "name": "supervisor_agent",
            "description": "A multi-agent supervisor system routing queries to specialized workers (RAG, Web, Graph, Stats).",
            "type": "supervisor",
            "workers": ["rag_worker", "web_worker", "graph_worker", "stats_worker"]
        }
