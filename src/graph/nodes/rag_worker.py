from typing import Dict, Any, List
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from src.core.settings import settings
from src.graph.state import AgentState
from src.knowledge.store.vector import VectorStore

class RagWorker:
    def __init__(self):
        # Initialize LLM
        self.llm = ChatOpenAI(
            model=settings.llm.model_name,
            api_key=settings.llm.api_key,
            base_url=settings.llm.api_base,
            temperature=0.3
        )
        # Initialize Vector Store (Lazy load possible, but init here for now)
        self.vector_store = VectorStore(
            collection_name=settings.database.milvus_collection_name or "pokemon_knowledge",
            embedding_model=settings.embedding.model_name, # managed by VectorStore internal resolver
            reranker_model=settings.embedding.reranker_model_name if settings.features.enable_reranker else None
        )

    def retrieve(self, query: str) -> str:
        try:
            results = self.vector_store.search(query, top_k=5, rerank=settings.features.enable_reranker)
            if not results:
                return "No relevant information found in the knowledge base."
            
            context = "\n\n".join([f"[{i+1}] {doc.page_content}" for i, doc in enumerate(results)])
            return context
        except Exception as e:
            return f"Error retrieving knowledge: {e}"

    def __call__(self, state: AgentState) -> Dict[str, Any]:
        """
        Worker node entry point.
        """
        messages = state["messages"]
        last_message = messages[-1]
        query = last_message.content
        
        # 1. Retrieve
        context = self.retrieve(query)
        
        # 2. Generate
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful Pokemon assistant. Use the following context to answer the user's question.\n\nContext:\n{context}"),
            ("user", "{query}")
        ])
        
        chain = prompt | self.llm
        response = chain.invoke({"context": context, "query": query})
        
        return {"messages": [response]}

# Factory for node
def rag_worker_node(state: AgentState) -> Dict[str, Any]:
    worker = RagWorker()
    return worker(state)
