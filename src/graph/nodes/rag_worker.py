import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from langchain_core.prompts import ChatPromptTemplate

from src.core.feature_flags import feature_enabled
from src.core.llm_factory import build_chat_llm
from src.core.settings import settings
from src.graph.nodes.crag import get_crag_evaluator
from src.graph.state import AgentState
from src.knowledge.core.operators import HyDEOperator
from src.knowledge.core.query_decomposer import get_query_decomposer
from src.knowledge.core.self_rag import get_self_rag
from src.knowledge.store.vector import VectorStore
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _adaptive_top_k(query: str, sub_query_count: int) -> int:
    """Determine top_k based on query complexity."""
    if sub_query_count > 1:
        return 8  # Complex multi-part queries need more docs
    if len(query) > 100:
        return 6  # Long queries may need broader context
    return 5  # Default for simple queries


class RagWorker:
    def __init__(self):
        # Initialize LLM
        self.llm = build_chat_llm(temperature=0.3)
        # Initialize Vector Store (Lazy load possible, but init here for now)
        self.vector_store = VectorStore(
            collection_name=settings.database.milvus_collection_name or "pokemon_knowledge",
            embedding_model=settings.embedding.model_name,  # managed by VectorStore internal resolver
            reranker_model=settings.reranker.model_name if feature_enabled("enable_reranker") else None,
        )

    def _retrieve_from_kb(self, query: str, db_id: str) -> str:
        if not feature_enabled("enable_knowledge_base"):
            return "Knowledge base is disabled."
        try:
            from src.knowledge.store.knowledgebase import KnowledgeBase

            kb = KnowledgeBase()
            kb_res = kb.search(
                query=query,
                db_id=db_id,
                rerank=feature_enabled("enable_reranker"),
                top_k=5,
            )
            results = kb_res.get("results") or []
            if not results:
                return "No relevant information found in the selected knowledge base."
            context_lines = []
            for i, item in enumerate(results, start=1):
                entity = item.get("entity") or {}
                text = entity.get("text") or ""
                if not text:
                    continue
                context_lines.append(f"[KB {i}] {text}")
            return "\n\n".join(context_lines) if context_lines else "No relevant information found."
        except Exception as e:
            return f"Error retrieving knowledge base: {e}"

    def retrieve(self, query: str, top_k: int = 5) -> str:
        try:
            # Use configured distance threshold for quality filtering
            threshold = getattr(settings.kb_config, "default_distance_threshold", 0.0)
            results = self.vector_store.search(
                query,
                top_k=top_k,
                rerank=feature_enabled("enable_reranker"),
                score_threshold=threshold,
            )
            if not results:
                return "No relevant information found in the knowledge base."

            # Context Window Expansion
            expanded_docs = []
            processed_ids = set()  # (file_id, chunk_index)

            for doc in results:
                meta = doc.metadata
                file_id = meta.get("file_id")
                idx = meta.get("chunk_index")

                if file_id is not None and idx is not None:
                    # Fetch window
                    window = self.vector_store.get_adjacent_chunks(file_id, idx, radius=1)
                    for w_doc in window:
                        w_meta = w_doc.metadata
                        tag = (w_meta.get("file_id"), w_meta.get("chunk_index"))
                        if tag not in processed_ids:
                            expanded_docs.append(w_doc)
                            processed_ids.add(tag)
                else:
                    # Fallback for docs without tracking info
                    if doc.page_content not in [d.page_content for d in expanded_docs]:
                        expanded_docs.append(doc)

            if not expanded_docs:
                return "No relevant information found."

            context = "\n\n".join([f"[{i + 1}] {doc.page_content}" for i, doc in enumerate(expanded_docs)])
            return context
        except Exception as e:
            return f"Error retrieving knowledge: {e}"

    def retrieve_with_crag(self, query: str, top_k: int = 5) -> str:
        """
        Retrieve with CRAG (Corrective RAG) - evaluates and corrects retrieval quality.
        """
        # 1. Initial retrieval
        threshold = getattr(settings.kb_config, "default_distance_threshold", 0.0)
        results = self.vector_store.search(
            query, top_k=top_k, rerank=feature_enabled("enable_reranker"), score_threshold=threshold,
        )

        if not results:
            # No docs - use web search directly
            logger.info("CRAG: No docs retrieved, falling back to web search")
            return self._web_search_context(query)

        # 2. Grade retrieval quality
        doc_contents = [doc.page_content for doc in results]
        evaluator = get_crag_evaluator()
        grade = evaluator.grade(query, doc_contents)

        # 3. Apply correction
        if grade.grade == "CORRECT":
            logger.info("CRAG: Retrieval CORRECT, using original docs")
            return self._format_context(results)

        elif grade.grade == "AMBIGUOUS":
            logger.info("CRAG: Retrieval AMBIGUOUS, supplementing with web")
            web_context = self._web_search_context(query)
            if web_context:
                return self._format_context(results) + "\n\n[Web Search Results]\n" + web_context
            return self._format_context(results)

        else:  # WRONG
            logger.info("CRAG: Retrieval WRONG, using web search only")
            web_context = self._web_search_context(query)
            if web_context:
                return web_context
            # Fallback to original if web fails
            return self._format_context(results)

    def _format_context(self, docs) -> str:
        """Format documents into context string."""
        return "\n\n".join([f"[{i + 1}] {doc.page_content}" for i, doc in enumerate(docs)])

    def _web_search_context(self, query: str) -> str:
        """Perform web search and return context."""
        try:
            from tavily import TavilyClient

            api_key = (settings.tavily.api_key or "").strip()
            if not api_key:
                return ""
            client = TavilyClient(api_key=api_key)
            response = client.search(query, max_results=3)
            if response and "results" in response:
                return "\n\n".join([f"[Web {i + 1}] {r.get('content', '')}" for i, r in enumerate(response["results"])])
        except Exception as e:
            logger.warning(f"Web search failed: {e}")
        return ""

    def __call__(self, state: AgentState) -> dict[str, Any]:
        """
        Worker node entry point.
        """
        messages = state["messages"]
        last_message = messages[-1]
        query = last_message.content

        # -1. Self-RAG (optional): Decide if retrieval is needed.
        # Keep this OFF by default to make the worker deterministic/offline-safe in tests.
        use_self_rag = bool(getattr(settings.features, "enable_self_rag", False))
        if use_self_rag:
            self_rag = get_self_rag()
            retrieval_decision = self_rag.should_retrieve(query)

            if not retrieval_decision.should_retrieve:
                # Skip retrieval, generate directly from LLM knowledge
                logger.info(f"Self-RAG: Skipping retrieval ({retrieval_decision.reason})")
                prompt = ChatPromptTemplate.from_messages(
                    [
                        ("system", "You are a helpful Pokemon assistant. Answer based on your knowledge."),
                        ("user", "{query}"),
                    ]
                )
                chain = prompt | self.llm
                response = chain.invoke({"query": query})
                return {"messages": [response]}

        db_id = state.get("db_id")
        if isinstance(db_id, str) and db_id.strip():
            context = self._retrieve_from_kb(query, db_id.strip())
        else:
            # 0. Query Decomposition for complex questions
            decomposer = get_query_decomposer()
            is_complex = decomposer.is_complex(query)
            if is_complex:
                sub_queries = decomposer.decompose(query)
                logger.info(f"Query decomposed into {len(sub_queries)} sub-queries")
            else:
                sub_queries = [query]

            # Adaptive top_k based on complexity
            top_k = _adaptive_top_k(query, len(sub_queries))

            # 1. Conditional HyDE - only for complex or long queries (saves LLM calls)
            use_hyde = is_complex or len(query) > 50
            hyde_query = query
            if use_hyde:
                try:
                    hyde_operator = HyDEOperator()

                    def model_wrapper(prompt_str):
                        return self.llm.invoke(prompt_str).content

                    hyde_doc = hyde_operator.call(model_wrapper, query, "")
                    hyde_query = hyde_doc
                    logger.debug("HyDE expansion applied")
                except Exception:
                    # Fallback to original query
                    pass

            # 2. Parallel sub-query retrieval for better performance
            all_contexts = []
            use_crag = feature_enabled("enable_web_search")

            def retrieve_single(sq: str) -> str:
                """Retrieve for a single sub-query."""
                # Use HyDE query only for the main query
                effective_query = hyde_query if sq == query else sq
                if use_crag:
                    return self.retrieve_with_crag(effective_query, top_k=top_k)
                return self.retrieve(effective_query, top_k=top_k)

            if len(sub_queries) > 1:
                # Parallel retrieval for multiple sub-queries
                with ThreadPoolExecutor(max_workers=min(len(sub_queries), 4)) as executor:
                    future_to_sq = {executor.submit(retrieve_single, sq): sq for sq in sub_queries}
                    for future in as_completed(future_to_sq):
                        sq = future_to_sq[future]
                        try:
                            ctx = future.result()
                            if ctx:
                                all_contexts.append(f"[Sub-Q: {sq[:50]}...]\n{ctx}")
                        except Exception as e:
                            logger.warning(f"Sub-query retrieval failed for '{sq[:30]}': {e}")
            else:
                # Single query - no parallelism needed
                ctx = retrieve_single(sub_queries[0])
                if ctx:
                    all_contexts.append(ctx)

            context = "\n\n---\n\n".join(all_contexts) if all_contexts else "No relevant information found."

        # 2. Generate
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful Pokemon assistant. Use the following context to answer the user's question.\n\nContext:\n{context}",
                ),
                ("user", "{query}"),
            ]
        )

        chain = prompt | self.llm
        response = chain.invoke({"context": context, "query": query})

        return {"messages": [response]}


# Factory for node
def rag_worker_node(state: AgentState) -> dict[str, Any]:
    worker = get_rag_worker()
    return worker(state)


_rag_worker: RagWorker | None = None


def get_rag_worker() -> RagWorker:
    """
    Cached worker instance.

    NOTE: tests patch classes heavily; avoid caching under pytest to keep patches effective.
    """
    if "pytest" in sys.modules:
        return RagWorker()
    global _rag_worker
    if _rag_worker is None:
        _rag_worker = RagWorker()
    return _rag_worker


def clear_rag_worker_cache() -> None:
    global _rag_worker
    _rag_worker = None
