from typing import Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from src.core.settings import settings
from src.graph.state import AgentState

# Placeholder for actual Neo4j logic
class GraphWorker:
    def __init__(self):
        self.llm = ChatOpenAI(
            model=settings.llm.model_name,
            api_key=settings.llm.api_key,
            base_url=settings.llm.api_base,
            temperature=0
        )

    def query_graph(self, query: str) -> str:
        # TODO: Implement Cypher generation and Neo4j execution
        # For now, return a mocked response saying graph is not ready
        return "Graph database lookup is not fully implemented yet."

    def __call__(self, state: AgentState) -> Dict[str, Any]:
        messages = state["messages"]
        last_message = messages[-1]
        query = last_message.content
        
        context = self.query_graph(query)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a graph database expert. Answer based on graph data.\n\nData:\n{context}"),
            ("user", "{query}")
        ])
        
        chain = prompt | self.llm
        response = chain.invoke({"context": context, "query": query})
        
        return {"messages": [response]}

def graph_worker_node(state: AgentState):
    worker = GraphWorker()
    return worker(state)
