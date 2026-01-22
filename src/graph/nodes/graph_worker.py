from typing import Dict, Any, List
from langchain_core.prompts.prompt import PromptTemplate
from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain
from langchain_openai import ChatOpenAI
from src.core.settings import settings
from src.graph.state import AgentState
from src.knowledge.store.graph import GraphStore
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Custom Cypher Generation Prompt
CYPHER_GENERATION_TEMPLATE = """Task:Generate Cypher statement to query a graph database.
Instructions:
Use only the provided relationship types and properties in the schema.
Do not use any other relationship types or properties that are not provided.
Schema:
{schema}
Note: Do not include any explanations or apologies in your responses.
Do not respond to any questions that might ask anything else than for you to construct a Cypher statement.
Do not include any text except the generated Cypher statement.

The question is:
{question}"""

CYPHER_GENERATION_PROMPT = PromptTemplate(
    input_variables=["schema", "question"], 
    template=CYPHER_GENERATION_TEMPLATE
)

class GraphWorker:
    def __init__(self):
        self.graph_store = GraphStore()
        
        self.llm = ChatOpenAI(
            model=settings.llm.model_name,
            api_key=settings.llm.api_key,
            base_url=settings.llm.api_base,
            temperature=0
        )
        
        if self.graph_store.graph:
            self.chain = GraphCypherQAChain.from_llm(
                self.llm,
                graph=self.graph_store.graph,
                verbose=True,
                cypher_prompt=CYPHER_GENERATION_PROMPT,
                allow_dangerous_requests=True,
                return_intermediate_steps=True
            )
        else:
            self.chain = None

    def __call__(self, state: AgentState) -> Dict[str, Any]:
        messages = state["messages"]
        last_message = messages[-1]
        query = last_message.content
        
        if not self.chain:
            response_text = "Graph database is not connected or enabled."
        else:
            try:
                # Invoke the chain
                response = self.chain.invoke({"query": query})
                response_text = response.get("result", "I couldn't find an answer in the graph.")
                
                # Extract graph data
                intermediate = response.get("intermediate_steps", [])
                if intermediate:
                    # intermediate_steps is tuple (str(cypher), List[Dict](context))
                    cypher_query = intermediate[0].get("query", "")
                    graph_data = intermediate[1] # This is the context/result
                    
                    # Serialize graph_data to JSON-Graph format
                    # Need to convert Neo4j objects (nodes/rels) to simple dicts
                    # Usually graph_data is a list of dicts like [{ 'p.name': 'Pikachu', ... }]
                    # We might need to construct nodes/edges.
                    
                    # For now, let's just dump the raw data into a json block 
                    # effectively giving the frontend the visual data
                    import json
                    
                    # Helper to serialize Neo4j types if they leak out, 
                    # but LangChain usually converts them to dicts/values?
                    # Let's hope it's serializable.
                    try:
                        graph_json = json.dumps(graph_data, default=str, ensure_ascii=False)
                        response_text += f"\n\n```json-graph\n{graph_json}\n```"
                        
                        # We also append the cypher for debugging
                        response_text += f"\n\n> **Cypher**: `{cypher_query}`"
                        
                    except Exception as json_err:
                        logger.warning(f"Failed to serialize graph data: {json_err}")

            except Exception as e:
                logger.error(f"GraphWorker query failed: {e}")
                response_text = f"Error querying knowledge graph: {str(e)}"
        
        # Wrap response in AIMessage
        from langchain_core.messages import AIMessage
        return {"messages": [AIMessage(content=response_text)]}

def graph_worker_node(state: AgentState):
    worker = GraphWorker()
    return worker(state)
