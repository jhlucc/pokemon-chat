"""
Semantic Cache for Pokemon Chat

Uses embedding similarity to cache and retrieve similar query responses.
Dramatically reduces LLM calls for repeated/similar questions.
"""
import json
import hashlib
from typing import Optional, Tuple, Dict, Any
from pathlib import Path
import numpy as np

from src.core.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

class SemanticCache:
    """
    Simple file-based semantic cache.
    
    For production, consider using Redis + pgvector or GPTCache.
    """
    
    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        similarity_threshold: float = 0.92,
        max_cache_size: int = 10000
    ):
        self.cache_dir = cache_dir or (settings.paths.data_dir / "semantic_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.similarity_threshold = similarity_threshold
        self.max_cache_size = max_cache_size
        
        # In-memory index: {query_hash: (embedding, response)}
        self._index: Dict[str, Tuple[np.ndarray, str]] = {}
        self._embedding_model = None
        
        self._load_cache()
    
    @property
    def embedding_model(self):
        """Lazy load embedding model."""
        if self._embedding_model is None:
            from langchain_openai import OpenAIEmbeddings
            self._embedding_model = OpenAIEmbeddings(
                model=settings.embedding.model_name,
                openai_api_key=settings.embedding.api_key,
                openai_api_base=settings.embedding.api_base,
            )
        return self._embedding_model
    
    def _get_query_hash(self, query: str) -> str:
        """Generate a hash for a query."""
        return hashlib.md5(query.strip().lower().encode()).hexdigest()
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    
    def _load_cache(self):
        """Load cache from disk."""
        cache_file = self.cache_dir / "cache.json"
        embeddings_file = self.cache_dir / "embeddings.npy"
        
        if cache_file.exists() and embeddings_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    cache_data = json.load(f)
                
                embeddings = np.load(embeddings_file, allow_pickle=True).item()
                
                for query_hash, response in cache_data.items():
                    if query_hash in embeddings:
                        self._index[query_hash] = (embeddings[query_hash], response)
                
                logger.info(f"Loaded {len(self._index)} cached entries")
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
    
    def _save_cache(self):
        """Save cache to disk."""
        cache_file = self.cache_dir / "cache.json"
        embeddings_file = self.cache_dir / "embeddings.npy"
        
        try:
            cache_data = {h: resp for h, (_, resp) in self._index.items()}
            embeddings = {h: emb for h, (emb, _) in self._index.items()}
            
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
            
            np.save(embeddings_file, embeddings)
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")
    
    def get(self, query: str) -> Optional[str]:
        """
        Try to get a cached response for a similar query.
        
        Returns None if no similar query is found.
        """
        if not self._index:
            return None
        
        try:
            # Get query embedding
            query_embedding = np.array(self.embedding_model.embed_query(query))
            
            # Find most similar cached query
            best_match = None
            best_similarity = 0.0
            
            for query_hash, (cached_embedding, response) in self._index.items():
                similarity = self._cosine_similarity(query_embedding, cached_embedding)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = response
            
            if best_similarity >= self.similarity_threshold:
                logger.info(f"Cache HIT: similarity={best_similarity:.3f}")
                return best_match
            else:
                logger.debug(f"Cache MISS: best similarity={best_similarity:.3f}")
                return None
                
        except Exception as e:
            logger.error(f"Cache lookup failed: {e}")
            return None
    
    def set(self, query: str, response: str):
        """
        Cache a query-response pair.
        """
        try:
            query_hash = self._get_query_hash(query)
            
            # Check if already exists
            if query_hash in self._index:
                return
            
            # Enforce max size (LRU-like: remove oldest)
            if len(self._index) >= self.max_cache_size:
                oldest_key = next(iter(self._index))
                del self._index[oldest_key]
            
            # Get embedding and store
            embedding = np.array(self.embedding_model.embed_query(query))
            self._index[query_hash] = (embedding, response)
            
            # Persist to disk
            self._save_cache()
            
            logger.debug(f"Cached new entry: {query[:50]}...")
            
        except Exception as e:
            logger.error(f"Failed to cache response: {e}")
    
    def clear(self):
        """Clear all cached entries."""
        self._index.clear()
        self._save_cache()
        logger.info("Cache cleared")


# Global instance
_cache_instance: Optional[SemanticCache] = None

def get_semantic_cache() -> SemanticCache:
    """Get the global semantic cache instance."""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = SemanticCache()
    return _cache_instance
