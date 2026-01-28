import sys
from typing import Any

from langchain_core.prompts import ChatPromptTemplate

from src.core.llm_factory import build_chat_llm
from src.graph.state import AgentState


class StatsWorker:
    def __init__(self):
        self.llm = build_chat_llm(temperature=0.0)

    def analyze(self, query: str) -> str:
        # TODO: Implement structured data analysis (e.g. Pandas/SQL)
        return "Detailed statistical analysis is pending implementation."

    def __call__(self, state: AgentState) -> dict[str, Any]:
        messages = state["messages"]
        last_message = messages[-1]
        query = last_message.content

        context = self.analyze(query)

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a data analyst. Provide insights based on data.\n\nAnalysis:\n{context}"),
                ("user", "{query}"),
            ]
        )

        chain = prompt | self.llm
        response = chain.invoke({"context": context, "query": query})

        return {"messages": [response]}


def stats_worker_node(state: AgentState):
    worker = get_stats_worker()
    return worker(state)


_stats_worker: StatsWorker | None = None


def get_stats_worker() -> StatsWorker:
    """
    Cached worker instance.

    NOTE: tests patch classes heavily; avoid caching under pytest to keep patches effective.
    """
    if "pytest" in sys.modules:
        return StatsWorker()
    global _stats_worker
    if _stats_worker is None:
        _stats_worker = StatsWorker()
    return _stats_worker


def clear_stats_worker_cache() -> None:
    global _stats_worker
    _stats_worker = None
