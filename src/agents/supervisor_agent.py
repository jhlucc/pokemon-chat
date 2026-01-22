from typing import Dict, Any, Optional
import os
import sqlite3
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.sqlite import SqliteSaver
from src.agents.base import BaseAgent
from src.graph.workflow import workflow
from src.core.settings import settings
from src.utils.callbacks import FileTraceCallbackHandler

class SupervisorAgent(BaseAgent):
    """
    Supervisor Agent that uses the modular LangGraph workflow.
    """
    def _build_graph(self) -> CompiledStateGraph:
        """
        Builds the graph using the pre-defined workflow and the agent's checkpointer.
        Uses SqliteSaver for persistence if configured.
        """
        # Ensure directory exists
        db_path = os.path.join(settings.paths.save_yaml_path, "agent_checkpoints.sqlite")
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # We need to manage the connection context manually or let SqliteSaver handle it.
        # SqliteSaver.from_conn_string(conn_string) is cleaner.
        # But for concurrent access in typical python app, simple connection is okay for now.
        conn = sqlite3.connect(db_path, check_same_thread=False)
        self._checkpointer = SqliteSaver(conn)
        
        # Attach tracing callback by updating the compiled graph's runtime config?
        # LangGraph invoke passes config. We can add callbacks there.
        # BaseAgent.invoke methods should support passing config.
        # But we can also set default config here? No, better in BaseAgent.
        
        return workflow.compile(checkpointer=self._checkpointer)

    def invoke(self, input: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Override to inject callbacks by default."""
        config = config or {}
        if "callbacks" not in config:
             # Enable Tracing
             config["callbacks"] = [FileTraceCallbackHandler()]
        return super().invoke(input, config)

    def get_info(self) -> dict:
        return {
            "name": "supervisor_agent",
            "description": "A multi-agent supervisor system routing queries to specialized workers (RAG, Web, Graph, Stats).",
            "type": "supervisor",
            "workers": ["rag_worker", "web_worker", "graph_worker", "stats_worker"]
        }
