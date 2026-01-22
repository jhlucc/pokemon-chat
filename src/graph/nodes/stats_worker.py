from typing import Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from src.core.settings import settings
from src.graph.state import AgentState

class StatsWorker:
    def __init__(self):
        self.llm = ChatOpenAI(
            model=settings.llm.model_name,
            api_key=settings.llm.api_key,
            base_url=settings.llm.api_base,
            temperature=0
        )

    def analyze(self, query: str) -> str:
        # TODO: Implement structured data analysis (e.g. Pandas/SQL)
        return "Detailed statistical analysis is pending implementation."

    def __call__(self, state: AgentState) -> Dict[str, Any]:
        messages = state["messages"]
        last_message = messages[-1]
        query = last_message.content
        
        context = self.analyze(query)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a data analyst. Provide insights based on data.\n\nAnalysis:\n{context}"),
            ("user", "{query}")
        ])
        
        chain = prompt | self.llm
        response = chain.invoke({"context": context, "query": query})
        
        return {"messages": [response]}

def stats_worker_node(state: AgentState):
    worker = StatsWorker()
    return worker(state)
