"""
Tests for Phase 12: Advanced RAG Techniques
- Speculative RAG
- Self-RAG
- Knowledge Refresh
"""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest


class TestSpeculativeRAG:
    """Test Speculative RAG functionality."""

    def test_draft_answer_model(self):
        from src.knowledge.core.speculative_rag import DraftAnswer

        draft = DraftAnswer(
            answer="Pikachu is an Electric-type Pokemon",
            confidence=0.9,
            reasoning="Clear factual question about Pokemon type",
        )

        assert draft.confidence == 0.9
        assert "Electric" in draft.answer

    def test_verification_result_model(self):
        from src.knowledge.core.speculative_rag import VerificationResult

        result = VerificationResult(
            selected_index=1,
            final_answer="Pikachu is an Electric-type Pokemon known for its Thunderbolt attack.",
            verification_notes="Draft 1 was most accurate and complete",
        )

        assert result.selected_index == 1
        assert "Thunderbolt" in result.final_answer


class TestSelfRAG:
    """Test Self-RAG functionality."""

    def test_retrieval_decision_model(self):
        from src.knowledge.core.self_rag import RetrievalDecision

        decision = RetrievalDecision(
            should_retrieve=True, reason="Query asks for specific Pokemon stats", query_type="factual"
        )

        assert decision.should_retrieve is True
        assert decision.query_type == "factual"

    def test_relevance_check_model(self):
        from src.knowledge.core.self_rag import RelevanceCheck

        check = RelevanceCheck(
            is_relevant=True, relevance_score=0.85, critique="Documents directly address the Pokemon types query"
        )

        assert check.relevance_score == 0.85

    def test_support_check_model(self):
        from src.knowledge.core.self_rag import SupportCheck

        check = SupportCheck(is_supported=True, support_level="full", unsupported_claims="")

        assert check.support_level == "full"


class TestKnowledgeRefresh:
    """Test Knowledge Refresh functionality."""

    def test_manager_initialization(self):
        with patch("src.knowledge.core.refresh.settings") as mock_settings:
            mock_settings.paths.data_dir = Path(tempfile.mkdtemp())

            from src.knowledge.core.refresh import KnowledgeRefreshManager

            manager = KnowledgeRefreshManager(db_path=mock_settings.paths.data_dir / "test_refresh.db")

            # Check DB was created
            assert manager.db_path.exists()

    def test_content_hash(self):
        with patch("src.knowledge.core.refresh.settings") as mock_settings:
            mock_settings.paths.data_dir = Path(tempfile.mkdtemp())

            from src.knowledge.core.refresh import KnowledgeRefreshManager

            manager = KnowledgeRefreshManager(db_path=mock_settings.paths.data_dir / "test_refresh.db")

            hash1 = manager._compute_hash("Hello World")
            hash2 = manager._compute_hash("Hello World")
            hash3 = manager._compute_hash("Different Content")

            assert hash1 == hash2
            assert hash1 != hash3

    def test_chunk_text(self):
        with patch("src.knowledge.core.refresh.settings") as mock_settings:
            mock_settings.paths.data_dir = Path(tempfile.mkdtemp())

            from src.knowledge.core.refresh import KnowledgeRefreshManager

            manager = KnowledgeRefreshManager(db_path=mock_settings.paths.data_dir / "test_refresh.db")

            text = "This is a sample text " * 100  # Long text
            chunks = manager._chunk_text(text, chunk_size=100)

            assert len(chunks) > 1
            assert all(len(chunk) <= 150 for chunk in chunks)  # Allow some overflow


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
