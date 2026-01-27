"""
Semantic Cache for Pokemon Chat

Uses embedding similarity to cache and retrieve similar query responses.
Dramatically reduces LLM calls for repeated/similar questions.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np

from src.core.provider_config import get_provider_api_base, get_provider_api_key
from src.core.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

_EMBEDDINGS_SUFFIX = "/embeddings"


def _normalize_openai_api_base(url: str) -> str:
    """
    Normalize OpenAI-compatible base URL for LangChain's OpenAIEmbeddings.

    Users sometimes configure `embedding_api_base` as the full embeddings endpoint
    (e.g. `.../v1/embeddings`). LangChain expects the *API base* (e.g. `.../v1`).
    """
    u = (url or "").strip().rstrip("/")
    if u.lower().endswith(_EMBEDDINGS_SUFFIX):
        u = u[: -len(_EMBEDDINGS_SUFFIX)]
    return u.rstrip("/")


@dataclass(frozen=True)
class _CacheEntry:
    embedding: np.ndarray
    response: str
    created_at_s: float
    meta: dict[str, str]


class SemanticCache:
    """
    Simple file-based semantic cache.

    For production, consider using Redis + pgvector or GPTCache.
    """

    def __init__(
        self,
        cache_dir: Path | None = None,
        similarity_threshold: float = 0.92,
        max_cache_size: int = 10000,
        ttl_seconds: int = 7 * 24 * 3600,
    ):
        # Isolate caches per embedding config to avoid cross-model poisoning.
        # This also means you can safely change embedding models without having to clear old caches.
        if cache_dir is None:
            provider = (settings.embedding.provider or "").strip().lower()
            model_name = (settings.embedding.model_name or "").strip()
            api_base = _normalize_openai_api_base(
                (settings.embedding.api_base or "").strip() or (get_provider_api_base(provider) or "").strip()
            )
            ns = hashlib.sha256(f"{provider}:{model_name}:{api_base}".encode()).hexdigest()[:12]
            cache_dir = settings.paths.cache_dir / "semantic_cache" / ns

        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.similarity_threshold = similarity_threshold
        self.max_cache_size = max_cache_size
        self.ttl_seconds = int(ttl_seconds)

        # In-memory index: {query_hash: (embedding, response)}
        self._index: dict[str, _CacheEntry] = {}
        self._embedding_model = None
        self._embedding_disabled = False
        self._lock = Lock()

        self._load_cache()

    @property
    def embedding_model(self):
        """Lazy load embedding model."""
        if self._embedding_disabled:
            return None

        if self._embedding_model is None:
            from langchain_openai import OpenAIEmbeddings

            provider = (settings.embedding.provider or "").strip().lower()
            api_key = (
                (settings.embedding.api_key or "").strip()
                or (get_provider_api_key(provider) or "").strip()
                or (settings.get_api_key(provider) or "").strip()
                or (settings.llm.api_key or "").strip()
            )
            api_base = _normalize_openai_api_base(
                (settings.embedding.api_base or "").strip()
                or (get_provider_api_base(provider) or "").strip()
                or (settings.llm.api_base or "").strip()
            )

            if not api_key:
                self._embedding_disabled = True
                logger.warning("Semantic cache disabled: embedding api_key is empty.")
                return None

            self._embedding_model = OpenAIEmbeddings(
                model=settings.embedding.model_name,
                openai_api_key=api_key,
                openai_api_base=api_base,
            )
        return self._embedding_model

    def _get_query_hash(self, query: str) -> str:
        return hashlib.md5(query.strip().lower().encode()).hexdigest()

    def _normalize_meta(self, meta: dict[str, Any] | None) -> dict[str, str]:
        """
        Normalize cache meta so comparisons are stable and keys remain small.

        We only keep a small allowlist to prevent the cache key from exploding.
        """
        meta = meta or {}
        out: dict[str, str] = {}
        for k in ("model_provider", "model_name", "system_prompt_sha"):
            v = meta.get(k, None)
            if v is None:
                continue
            s = str(v).strip()
            if s:
                out[k] = s
        return out

    def _meta_key(self, meta: dict[str, Any] | None) -> str:
        norm = self._normalize_meta(meta)
        if not norm:
            return ""
        # Stable serialization for hashing & comparisons
        return json.dumps(norm, sort_keys=True, ensure_ascii=False, separators=(",", ":"))

    def _get_entry_hash(self, query: str, meta: dict[str, Any] | None) -> str:
        """
        Hash for a cache entry.

        Backward compatible:
        - If meta is empty, this is identical to the legacy query hash.
        - If meta is present, we isolate entries per meta namespace.
        """
        mk = self._meta_key(meta)
        if not mk:
            return self._get_query_hash(query)
        return hashlib.md5(f"{mk}::{query.strip().lower()}".encode()).hexdigest()

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0.0:
            return 0.0
        return float(np.dot(a, b) / denom)

    def _load_cache(self):
        """Load cache from disk."""
        cache_file = self.cache_dir / "cache.json"
        embeddings_file = self.cache_dir / "embeddings.npy"

        if cache_file.exists() and embeddings_file.exists():
            try:
                with open(cache_file, encoding="utf-8") as f:
                    cache_data = json.load(f)

                embeddings = np.load(embeddings_file, allow_pickle=True).item()

                now = time.time()
                loaded = 0
                for query_hash, payload in cache_data.items():
                    if query_hash not in embeddings:
                        continue

                    # Backward compatible format:
                    # - old: {hash: "response"}
                    # - new: {hash: {"response": "...", "created_at_s": 123.4, "meta": {...}}}
                    if isinstance(payload, str):
                        response = payload
                        created_at_s = 0.0
                        meta = {}
                    elif isinstance(payload, dict):
                        response = str(payload.get("response", ""))
                        created_at_s = float(payload.get("created_at_s", 0.0))
                        meta = payload.get("meta") or {}
                        if not isinstance(meta, dict):
                            meta = {}
                        # Keep storage normalized for comparisons.
                        meta = self._normalize_meta(meta)
                    else:
                        continue

                    # TTL eviction on load (best-effort).
                    if self.ttl_seconds > 0 and created_at_s and (now - created_at_s) > self.ttl_seconds:
                        continue

                    self._index[query_hash] = _CacheEntry(
                        embedding=np.array(embeddings[query_hash]),
                        response=response,
                        created_at_s=created_at_s or now,
                        meta=meta,
                    )
                    loaded += 1

                logger.info(f"Loaded {loaded} cached entries")
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")

    def _save_cache(self):
        """Save cache to disk."""
        cache_file = self.cache_dir / "cache.json"
        embeddings_file = self.cache_dir / "embeddings.npy"
        cache_tmp = cache_file.with_suffix(cache_file.suffix + ".tmp")
        emb_tmp = embeddings_file.with_suffix(embeddings_file.suffix + ".tmp")

        try:
            cache_data = {
                h: {"response": entry.response, "created_at_s": entry.created_at_s, "meta": entry.meta}
                for h, entry in self._index.items()
            }
            embeddings = {h: entry.embedding for h, entry in self._index.items()}

            # Atomic-ish writes to avoid corrupting on process crash.
            cache_tmp.write_text(json.dumps(cache_data, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(cache_tmp, cache_file)

            # NOTE: np.save(path) appends ".npy" if the filename doesn't end with ".npy".
            # Write via file object so our temp filename can be anything.
            with open(emb_tmp, "wb") as f:
                np.save(f, embeddings)
            os.replace(emb_tmp, embeddings_file)
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")
            # Best-effort cleanup
            try:
                if cache_tmp.exists():
                    cache_tmp.unlink()
            except Exception:
                pass
            try:
                if emb_tmp.exists():
                    emb_tmp.unlink()
            except Exception:
                pass

    def get(self, query: str, meta: dict[str, Any] | None = None) -> str | None:
        """
        Try to get a cached response for a similar query.

        Returns None if no similar query is found.
        """
        with self._lock:
            if not self._index:
                return None

        try:
            model = self.embedding_model
            if model is None:
                return None

            # Only match within the same meta namespace.
            norm_meta = self._normalize_meta(meta)

            # Fast path: exact match (no embedding call).
            now = time.time()
            query_hash = self._get_entry_hash(query, norm_meta)
            with self._lock:
                entry = self._index.get(query_hash)
                if entry is not None and entry.meta == norm_meta:
                    if self.ttl_seconds > 0 and entry.created_at_s and (now - entry.created_at_s) > self.ttl_seconds:
                        self._index.pop(query_hash, None)
                    else:
                        # LRU bump
                        self._index.pop(query_hash, None)
                        self._index[query_hash] = entry
                        logger.info("Cache HIT (exact)")
                        return entry.response

            # Get query embedding
            query_embedding = np.array(model.embed_query(query))

            # Find most similar cached query
            best_hash: str | None = None
            best_match = None
            best_similarity = 0.0

            with self._lock:
                # Drop expired entries opportunistically.
                if self.ttl_seconds > 0:
                    expired = [
                        h
                        for h, entry in self._index.items()
                        if entry.created_at_s and (now - entry.created_at_s) > self.ttl_seconds
                    ]
                    for h in expired:
                        self._index.pop(h, None)

                items = list(self._index.items())

            for h, entry in items:
                if entry.meta != norm_meta:
                    continue
                similarity = self._cosine_similarity(query_embedding, entry.embedding)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_hash = h
                    best_match = entry.response

            if best_similarity >= self.similarity_threshold:
                logger.info(f"Cache HIT: similarity={best_similarity:.3f}")
                # LRU bump for the matched entry.
                if best_hash:
                    with self._lock:
                        e = self._index.pop(best_hash, None)
                        if e is not None:
                            self._index[best_hash] = e
                return best_match
            else:
                logger.debug(f"Cache MISS: best similarity={best_similarity:.3f}")
                return None

        except Exception as e:
            logger.error(f"Cache lookup failed: {e}")
            return None

    def set(self, query: str, response: str, meta: dict[str, Any] | None = None):
        """
        Cache a query-response pair.
        """
        try:
            model = self.embedding_model
            if model is None:
                return
            norm_meta = self._normalize_meta(meta)
            query_hash = self._get_entry_hash(query, norm_meta)

            # Check if already exists
            with self._lock:
                if query_hash in self._index:
                    # LRU bump
                    entry = self._index.pop(query_hash)
                    self._index[query_hash] = entry
                    return

            # Enforce max size (LRU-like: remove oldest)
            with self._lock:
                if len(self._index) >= self.max_cache_size:
                    oldest_key = next(iter(self._index))
                    del self._index[oldest_key]

            # Get embedding and store
            embedding = np.array(model.embed_query(query))
            with self._lock:
                self._index[query_hash] = _CacheEntry(
                    embedding=embedding,
                    response=response,
                    created_at_s=time.time(),
                    meta=norm_meta,
                )

            # Persist to disk
            with self._lock:
                self._save_cache()

            logger.debug(f"Cached new entry: {query[:50]}...")

        except Exception as e:
            logger.error(f"Failed to cache response: {e}")

    def clear(self):
        """Clear all cached entries."""
        with self._lock:
            self._index.clear()
            self._save_cache()
        logger.info("Cache cleared")

    def stats(self) -> dict[str, Any]:
        """Return lightweight cache stats for admin/debug endpoints."""
        with self._lock:
            size = len(self._index)
        return {
            "size": size,
            "max_cache_size": self.max_cache_size,
            "ttl_seconds": self.ttl_seconds,
            "similarity_threshold": self.similarity_threshold,
            "cache_dir": str(self.cache_dir),
            "embedding_disabled": bool(self._embedding_disabled),
        }


# Global instance
_cache_instance: SemanticCache | None = None


def get_semantic_cache() -> SemanticCache:
    """Get the global semantic cache instance."""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = SemanticCache()
    return _cache_instance
