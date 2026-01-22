from typing import Dict, Any, List
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from src.core.settings import settings
from src.graph.state import AgentState
from src.graph.state import AgentState
from src.knowledge.store.vector import VectorStore
from src.knowledge.core.operators import HyDEOperator

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
            reranker_model=settings.reranker.model_name if settings.features.enable_reranker else None
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
        
        # 1. HyDE Expansion (Optional but recommended)
        hyde_query = query
        try:
             # Basic HyDE: Generate a hypothetical answer
             hyde_operator = HyDEOperator()
             # We need a callable for the model, simplest is using self.llm.invoke
             # HyDEOperator expects a callable that takes a prompt string.
             # ChatOpenAI.invoke takes string or messages.
             # We wrap it.
             def model_wrapper(prompt_str):
                 return self.llm.invoke(prompt_str).content
                 
             hyde_doc = hyde_operator.call(model_wrapper, query, "")
             # Extend query with hypothetical document for retrieval? 
             # Standard HyDE uses the vector of the hypothetical doc.
             # Our VectorStore.search takes a string and embeds it.
             # So we can pass the hypothetical doc as the query string.
             hyde_query = hyde_doc
        except Exception as e:
             # Fallback to original query
             pass

        # 2. Retrieve
        context = self.retrieve(hyde_query)
        
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
