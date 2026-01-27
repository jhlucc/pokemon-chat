"""
Unit tests for the file-based SemanticCache.

These tests must be offline-safe: we stub the embedding model to avoid network calls.
"""

from __future__ import annotations


class _DummyEmbeddings:
    def __init__(self, vectors: dict[str, list[float]]):
        self._vectors = vectors

    def embed_query(self, query: str) -> list[float]:
        return self._vectors[query]


def test_semantic_cache_set_get_roundtrip(tmp_path):
    from src.knowledge.cache.cache import SemanticCache

    cache = SemanticCache(cache_dir=tmp_path, similarity_threshold=0.95, ttl_seconds=3600)
    cache._embedding_model = _DummyEmbeddings({"pikachu": [1.0, 0.0]})  # noqa: SLF001 (test stub)

    cache.set("pikachu", "Pikachu is an Electric-type Pokemon.")
    assert cache.get("pikachu") == "Pikachu is an Electric-type Pokemon."


def test_semantic_cache_ttl_expiry(tmp_path, monkeypatch):
    import src.knowledge.cache.cache as cache_mod
    from src.knowledge.cache.cache import SemanticCache

    t = 1_000_000.0
    monkeypatch.setattr(cache_mod.time, "time", lambda: t)

    cache = SemanticCache(cache_dir=tmp_path, similarity_threshold=0.95, ttl_seconds=1)
    cache._embedding_model = _DummyEmbeddings({"q": [1.0, 0.0]})  # noqa: SLF001 (test stub)

    cache.set("q", "a")
    assert cache.get("q") == "a"

    t += 2.0
    assert cache.get("q") is None


def test_semantic_cache_persistence_reload(tmp_path):
    from src.knowledge.cache.cache import SemanticCache

    vecs = {"q": [0.0, 1.0]}

    cache1 = SemanticCache(cache_dir=tmp_path, similarity_threshold=0.5, ttl_seconds=3600)
    cache1._embedding_model = _DummyEmbeddings(vecs)  # noqa: SLF001 (test stub)
    cache1.set("q", "answer")

    # New instance should load from disk and still be able to match with the same embeddings.
    cache2 = SemanticCache(cache_dir=tmp_path, similarity_threshold=0.5, ttl_seconds=3600)
    cache2._embedding_model = _DummyEmbeddings(vecs)  # noqa: SLF001 (test stub)
    assert cache2.get("q") == "answer"


def test_semantic_cache_exact_match_does_not_require_embedding_call(tmp_path):
    from src.knowledge.cache.cache import SemanticCache

    cache = SemanticCache(cache_dir=tmp_path, similarity_threshold=0.95, ttl_seconds=3600)
    cache._embedding_model = _DummyEmbeddings({"pikachu": [1.0, 0.0]})  # noqa: SLF001 (test stub)
    cache.set("pikachu", "Pikachu is Electric.")

    class _ExplodingEmbeddings:
        def embed_query(self, query: str):
            raise AssertionError("embed_query should not be called for exact-match hits")

    # Exact-match path should return without calling embeddings.
    cache._embedding_model = _ExplodingEmbeddings()  # noqa: SLF001 (test stub)
    assert cache.get("pikachu") == "Pikachu is Electric."


def test_semantic_cache_lru_eviction(tmp_path):
    from src.knowledge.cache.cache import SemanticCache

    cache = SemanticCache(cache_dir=tmp_path, similarity_threshold=0.99, ttl_seconds=3600, max_cache_size=2)
    cache._embedding_model = _DummyEmbeddings(  # noqa: SLF001 (test stub)
        {"q1": [1.0, 0.0], "q2": [0.0, 1.0], "q3": [1.0, 1.0]}
    )

    cache.set("q1", "a1")
    cache.set("q2", "a2")
    assert cache.get("q1") == "a1"  # LRU bump: q1 becomes most-recent

    cache.set("q3", "a3")  # should evict q2 (least-recent)
    assert cache.get("q2") is None


def test_semantic_cache_meta_isolation(tmp_path):
    from src.knowledge.cache.cache import SemanticCache

    cache = SemanticCache(cache_dir=tmp_path, similarity_threshold=0.95, ttl_seconds=3600)
    cache._embedding_model = _DummyEmbeddings({"q": [1.0, 0.0]})  # noqa: SLF001 (test stub)

    meta_a = {"model_provider": "openai", "model_name": "gpt-4o", "system_prompt_sha": "aaa"}
    meta_b = {"model_provider": "openai", "model_name": "gpt-4o-mini", "system_prompt_sha": "bbb"}

    cache.set("q", "a1", meta=meta_a)
    cache.set("q", "a2", meta=meta_b)

    assert cache.get("q", meta=meta_a) == "a1"
    assert cache.get("q", meta=meta_b) == "a2"
    assert cache.get("q", meta={"model_provider": "openai", "model_name": "other", "system_prompt_sha": "ccc"}) is None


def test_settings_paths_have_cache_and_data_dir():
    from src.core.settings import settings

    assert settings.paths.data_dir.parts[-2:] == ("resources", "data")
    assert settings.paths.cache_dir.parts[-2:] == ("resources", "cache")
