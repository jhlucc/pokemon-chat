import sys
from typing import Any

from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain
from langchain_core.prompts.prompt import PromptTemplate

from src.core.llm_factory import build_chat_llm
from src.core.settings import settings
from src.graph.state import AgentState
from src.knowledge.store.graph import GraphStore
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Custom Cypher Generation Prompt with few-shot examples
CYPHER_GENERATION_TEMPLATE = """Task: Generate a Cypher statement to query a graph database.

Instructions:
- Use only the provided relationship types and properties in the schema.
- Always start with MATCH, then specify the pattern, then RETURN the results.
- Do not include any explanations, only output the Cypher query.

Schema:
{schema}

Examples:
Question: 皮卡丘进化是什么？
Cypher: MATCH (p:Pokémon {{name: "皮卡丘"}})-[:evolves_into]->(e:Pokémon) RETURN e.name AS evolution

Question: 小智有哪些宝可梦？
Cypher: MATCH (t:Person {{name: "小智"}})-[:has_pokemon]->(p:Pokémon) RETURN p.name AS pokemon

Question: 真新镇在哪个地区？
Cypher: MATCH (t:Town {{name: "真新镇"}})-[:located_in]->(r:Region) RETURN r.name AS region

Question: 皮卡丘是什么属性？
Cypher: MATCH (p:Pokémon {{name: "皮卡丘"}})-[:has_type]->(t:identity) RETURN t.name AS type

Now generate a Cypher query for this question:
Question: {question}
Cypher:"""

CYPHER_GENERATION_PROMPT = PromptTemplate(input_variables=["schema", "question"], template=CYPHER_GENERATION_TEMPLATE)


class GraphWorker:
    def __init__(self):
        self.graph_store = GraphStore()

        self.llm = build_chat_llm(temperature=0.0)

        if self.graph_store.graph:
            self.chain = GraphCypherQAChain.from_llm(
                self.llm,
                graph=self.graph_store.graph,
                verbose=True,
                cypher_prompt=CYPHER_GENERATION_PROMPT,
                allow_dangerous_requests=bool(getattr(settings.features, "allow_dangerous_graph_requests", False)),
                return_intermediate_steps=True,
            )
        else:
            self.chain = None

    def __call__(self, state: AgentState) -> dict[str, Any]:
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
                    graph_data = intermediate[1]  # This is the context/result

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
    worker = get_graph_worker()
    return worker(state)


_graph_worker: GraphWorker | None = None


def get_graph_worker() -> GraphWorker:
    """
    Cached worker instance.

    NOTE: tests patch classes heavily; avoid caching under pytest to keep patches effective.
    """
    if "pytest" in sys.modules:
        return GraphWorker()
    global _graph_worker
    if _graph_worker is None:
        _graph_worker = GraphWorker()
    return _graph_worker


def clear_graph_worker_cache() -> None:
    global _graph_worker
    _graph_worker = None
