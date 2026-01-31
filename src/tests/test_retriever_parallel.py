from __future__ import annotations

import time
from unittest.mock import patch

from src.knowledge.core.retriever import Retriever


def _sleepy(value: dict, delay_s: float):
    def _fn(_query, _history, _refs):  # noqa: ANN001
        time.sleep(delay_s)
        return value

    return _fn


def test_retriever_parallelizes_independent_substeps():
    # Avoid initializing optional runtime dependencies (reranker/websearch) in unit tests.
    with patch.object(Retriever, "_load_models", lambda self: None):
        retriever = Retriever()

    # Prevent any internal LLM calls from the entity recognizer path.
    retriever.reco_entities = lambda *_args, **_kwargs: []  # type: ignore[assignment]

    # Four independent steps at 0.25s each:
    # - sequential: ~1.00s
    # - parallel:   ~0.25-0.40s
    delay = 0.25
    retriever.query_knowledgebase = _sleepy({"kb": True}, delay)  # type: ignore[assignment]
    retriever.query_graph = _sleepy({"graph": True}, delay)  # type: ignore[assignment]
    retriever.query_web = _sleepy({"web": True}, delay)  # type: ignore[assignment]
    retriever.query_mysql_mcp = _sleepy({"mcp": True}, delay)  # type: ignore[assignment]

    # Meta indicates multiple retrieval sources are requested -> should run in parallel.
    meta = {"db_id": "kb_test", "use_graph": True, "use_web": True, "mcp_id": "default"}

    start = time.perf_counter()
    refs = retriever.retrieval(query="q", history=[], meta=meta)
    elapsed = time.perf_counter() - start

    assert refs["knowledge_base"] == {"kb": True}
    assert refs["graph_base"] == {"graph": True}
    assert refs["web_search"] == {"web": True}
    assert refs["mysql_mcp"] == {"mcp": True}

    # This is intentionally loose to avoid CI flakiness while still catching sequential execution.
    assert elapsed < 0.7, f"expected parallel execution (<0.7s), got {elapsed:.3f}s"
