from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from tavily import TavilyClient

from src.core.llm_factory import build_chat_llm
from src.core.settings import settings
from src.graph.state import AgentState


class WebWorker:
    def __init__(self):
        self.llm = build_chat_llm(temperature=0.5)
        api_key = (settings.tavily.api_key or "").strip()
        self.tavily = TavilyClient(api_key=api_key) if api_key else None

    def search(self, query: str) -> str:
        try:
            if self.tavily is None:
                return "Web search is not configured (tavily_api_key is empty)."
            # Simple Tavily search context
            response = self.tavily.search(query=query, search_depth="basic", max_results=3)
            results = response.get("results", [])
            if not results:
                return "No web results found."

            context = "\n\n".join(
                [f"Title: {r.get('title')}\nUrl: {r.get('url')}\nContent: {r.get('content')}" for r in results]
            )
            return context
        except Exception as e:
            return f"Web search failed: {str(e)}"

    def __call__(self, state: AgentState) -> dict[str, Any]:
        messages = state["messages"]
        last_message = messages[-1]
        query = last_message.content

        context = self.search(query)

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a web researcher. Answer the query using the provided search results.\n\nResults:\n{context}",
                ),
                ("user", "{query}"),
            ]
        )

        chain = prompt | self.llm
        response = chain.invoke({"context": context, "query": query})

        return {"messages": [response]}


def web_worker_node(state: AgentState):
    worker = WebWorker()
    return worker(state)
