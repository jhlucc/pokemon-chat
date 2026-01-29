import os
import sqlite3
from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.state import CompiledStateGraph

try:
    # Optional dependency: provided by `langgraph-checkpoint-sqlite`.
    from langgraph.checkpoint.sqlite import SqliteSaver  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    SqliteSaver = None
from src.agents.base import BaseAgent
from src.core.settings import settings
from src.graph.workflow import workflow
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
        type_ = (settings.agent.checkpointer_type or "memory").lower()

        if type_ == "sqlite" and SqliteSaver is not None:
            # Ensure directory exists
            db_path = os.path.join(settings.paths.save_yaml_path, "agent_checkpoints.sqlite")
            os.makedirs(os.path.dirname(db_path), exist_ok=True)

            # We need to manage the connection context manually or let SqliteSaver handle it.
            # SqliteSaver.from_conn_string(conn_string) is cleaner.
            # But for concurrent access in typical python app, simple connection is okay for now.
            conn = sqlite3.connect(db_path, check_same_thread=False)
            self._checkpointer = SqliteSaver(conn)
        else:
            # Fallback: in-memory checkpoints (works without extra deps).
            self._checkpointer = MemorySaver()

        # Attach tracing callback by updating the compiled graph's runtime config?
        # LangGraph invoke passes config. We can add callbacks there.
        # BaseAgent.invoke methods should support passing config.
        # But we can also set default config here? No, better in BaseAgent.

        return workflow.compile(checkpointer=self._checkpointer)

    def invoke(self, input: dict[str, Any], config: dict[str, Any] | None = None) -> dict[str, Any]:
        """Override to inject callbacks by default."""
        config = config or {}
        if "callbacks" not in config:
            # Enable Tracing
            config["callbacks"] = [FileTraceCallbackHandler()]
        return super().invoke(input, config)

    async def query(self, query: str, meta: dict[str, Any] | None = None, history: list[dict[str, Any]] | None = None):
        """
        Streaming query entrypoint used by `/chat/agent/{agent}`.

        This overrides BaseAgent.query so the frontend can constrain which workers the supervisor
        may route to (e.g. disable web/graph/mcp) via `meta.agent_constraints.allowed_workers`.
        """
        from langchain_core.messages import HumanMessage

        meta = meta or {}
        constraints = meta.get("agent_constraints") or {}
        allowed_workers = constraints.get("allowed_workers")
        if isinstance(allowed_workers, list):
            allowed_workers = [w for w in allowed_workers if isinstance(w, str)]
        else:
            allowed_workers = None

        input_state: dict[str, Any] = {"messages": [HumanMessage(content=query)]}
        if allowed_workers is not None:
            input_state["allowed_workers"] = allowed_workers
        db_id = meta.get("db_id")
        if isinstance(db_id, str) and db_id.strip():
            input_state["db_id"] = db_id.strip()

        config = {"configurable": {"thread_id": meta.get("thread_id", "default")}}

        async for event in self.graph.astream_events(input_state, config, version="v1"):
            kind = event["event"]

            if kind == "on_chat_model_stream":
                content = event["data"]["chunk"].content
                if content:
                    yield content
            elif kind == "on_tool_start":
                tool_name = event["name"]
                tool_input = event["data"].get("input")
                yield {
                    "status": "tool_start",
                    "status_text": f"正在使用工具 {tool_name}...",
                    "tool": tool_name,
                    "input": tool_input,
                }
            elif kind == "on_tool_end":
                tool_name = event["name"]
                yield {"status": "tool_end", "status_text": f"工具 {tool_name} 执行完成", "tool": tool_name}

    def get_info(self) -> dict:
        return {
            "name": "supervisor_agent",
            "description": "A multi-agent supervisor system routing queries to specialized workers (RAG, Web, Graph, Stats).",
            "type": "supervisor",
            "workers": ["rag_worker", "web_worker", "graph_worker", "stats_worker"],
        }
