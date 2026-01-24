"""
Unit tests for the file-based SemanticCache.

These tests must be offline-safe: we stub the embedding model to avoid network calls.
"""

from __future__ import annotations

from typing import Dict, List


class _DummyEmbeddings:
    def __init__(self, vectors: Dict[str, List[float]]):
        self._vectors = vectors

    def embed_query(self, query: str) -> List[float]:
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


def test_settings_paths_have_cache_and_data_dir():
    from src.core.settings import settings

    assert settings.paths.data_dir.parts[-2:] == ("resources", "data")
    assert settings.paths.cache_dir.parts[-2:] == ("resources", "cache")
