"""
Tests for Phase 10: Next-Gen Optimizations
- Semantic Caching
- CRAG Evaluator
- Query Decomposer
"""
import pytest
from unittest.mock import patch, MagicMock
import numpy as np


class TestSemanticCache:
    """Test Semantic Cache functionality."""
    
    def test_cache_initialization(self):
        with patch("src.knowledge.cache.semantic_cache.settings") as mock_settings:
            mock_settings.paths.data_dir = MagicMock()
            mock_settings.paths.data_dir.__truediv__ = MagicMock(return_value=MagicMock())
            
            from src.knowledge.cache.semantic_cache import SemanticCache
            cache = SemanticCache(cache_dir=MagicMock())
            
            assert cache.similarity_threshold == 0.92
            assert cache.max_cache_size == 10000
    
    def test_cosine_similarity(self):
        from src.knowledge.cache.semantic_cache import SemanticCache
        
        with patch("src.knowledge.cache.semantic_cache.settings"):
            cache = SemanticCache.__new__(SemanticCache)
            cache._index = {}
            
            # Test vectors
            a = np.array([1.0, 0.0, 0.0])
            b = np.array([1.0, 0.0, 0.0])
            
            similarity = cache._cosine_similarity(a, b)
            assert similarity == 1.0
            
            c = np.array([0.0, 1.0, 0.0])
            similarity_orth = cache._cosine_similarity(a, c)
            assert abs(similarity_orth) < 0.01


class TestCRAGEvaluator:
    """Test CRAG Evaluator functionality."""
    
    def test_grade_no_docs(self):
        with patch("src.graph.nodes.crag_evaluator.settings"):
            from src.graph.nodes.crag_evaluator import CRAGEvaluator, RetrievalGrade
            
            evaluator = CRAGEvaluator.__new__(CRAGEvaluator)
            evaluator.llm = MagicMock()
            evaluator.grading_chain = MagicMock()
            
            result = evaluator.grade("test query", [])
            
            assert result.grade == "WRONG"
            assert "No documents" in result.reason


class TestQueryDecomposer:
    """Test Query Decomposer functionality."""
    
    def test_is_complex_detection(self):
        with patch("src.knowledge.core.query_decomposer.settings"):
            from src.knowledge.core.query_decomposer import QueryDecomposer
            
            decomposer = QueryDecomposer.__new__(QueryDecomposer)
            
            # Complex queries (must contain exact keywords from is_complex)
            assert decomposer.is_complex("Compare Pikachu and Charmander") == True
            assert decomposer.is_complex("What is the difference between fire and water?") == True
            assert decomposer.is_complex("What's the difference between fire and water types?") == True
            
            # Simple queries
            assert decomposer.is_complex("Tell me about Pikachu") == False
            assert decomposer.is_complex("What is Pikachu's type?") == False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
