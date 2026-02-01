import sys
from typing import Any

from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain
from langchain_core.messages import AIMessage
from langchain_core.prompts.prompt import PromptTemplate

from src.agents.pokedex_shortcut import maybe_answer_pokedex
from src.agents.utils.message_filter import make_error_response, validate_worker_input
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
        # Lazy init: keep worker usable even when Neo4j/LLM aren't configured.
        self.llm = None
        self.chain = None

    def _ensure_chain(self) -> None:
        """Create the graph QA chain only when the graph is available."""
        if self.chain is not None:
            return
        if not self.graph_store.graph:
            return

        if self.llm is None:
            self.llm = build_chat_llm(temperature=0.0)

        self.chain = GraphCypherQAChain.from_llm(
            self.llm,
            graph=self.graph_store.graph,
            # Avoid leaking intermediate chain steps into user-facing streams.
            verbose=False,
            cypher_prompt=CYPHER_GENERATION_PROMPT,
            allow_dangerous_requests=bool(getattr(settings.features, "allow_dangerous_graph_requests", False)),
            return_intermediate_steps=False,
        )

    def __call__(self, state: AgentState) -> dict[str, Any]:
        # Validate input using shared utility
        query, error = validate_worker_input(state)
        if error:
            return make_error_response(error)

        # Prefer deterministic local Pokédex facts when the question is clearly a
        # Pokédex-style query (e.g. evolution/type/abilities). This avoids:
        # - Graph schema/encoding mismatches
        # - Missing edges for final-stage evolutions (e.g. 喷火龙)
        local = maybe_answer_pokedex(query)
        if local:
            return {"messages": [AIMessage(content=local.content)]}

        # If the graph isn't available, fall back to the local dataset for Pokédex facts
        # (prevents hard failures when Neo4j isn't running).
        if not self.graph_store.graph:
            # We already tried maybe_answer_pokedex above; if it didn't match,
            # this is likely a true graph-only query.
            return make_error_response(
                "知识图谱未连接或未启用，暂时无法查询图谱关系。你可以改问宝可梦图鉴类问题（属性/特性/进化等）。"
            )

        # Ensure chain (lazy).
        try:
            self._ensure_chain()
        except Exception as e:
            logger.warning(f"GraphWorker chain init failed: {e}")
            local = maybe_answer_pokedex(query)
            if local:
                return {"messages": [AIMessage(content=local.content)]}
            return make_error_response(f"知识图谱组件初始化失败: {e}")

        if not self.chain:
            return make_error_response("知识图谱未连接或未启用，暂时无法查询图谱关系。")

        try:
            # Invoke the chain
            response = self.chain.invoke({"query": query})
            response_text = response.get("result", "I couldn't find an answer in the graph.")

            # If the graph couldn't answer (common for final-stage evolution, missing edges, etc.),
            # fall back to the bundled Pokédex dataset.
            low = (response_text or "").strip().lower()
            if not (response_text or "").strip() or "couldn't find" in low or "could not find" in low:
                local = maybe_answer_pokedex(query)
                if local:
                    response_text = local.content

        except Exception as e:
            logger.error(f"GraphWorker query failed: {e}")
            local = maybe_answer_pokedex(query)
            if local:
                response_text = local.content
            else:
                response_text = f"查询知识图谱失败: {str(e)}"

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
