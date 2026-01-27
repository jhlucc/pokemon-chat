"""
Unit tests for core modules (settings, embedding, models)
"""

from unittest.mock import MagicMock, patch


class TestSettings:
    """Test settings module"""

    def test_settings_singleton(self):
        """Settings should be a singleton"""
        from src.core.settings import get_settings, settings

        settings1 = get_settings()
        settings2 = get_settings()
        assert settings1 is settings2
        assert settings is settings1

    def test_settings_has_required_fields(self):
        """Settings should have all required configuration fields"""
        from src.core.settings import settings

        # Check nested settings exist
        assert hasattr(settings, "paths")
        assert hasattr(settings, "database")
        assert hasattr(settings, "llm")
        assert hasattr(settings, "embedding")
        assert hasattr(settings, "reranker")
        assert hasattr(settings, "features")

    def test_get_api_key(self):
        """Test get_api_key method"""
        from src.core.settings import settings

        # Should return empty string for unknown provider
        result = settings.get_api_key("unknown_provider")
        assert result == ""


class TestEmbeddingCache:
    """Test embedding cache functionality"""

    def test_cache_stats(self):
        """Test cache stats method"""
        from src.models.embedding import BaseEmbeddingModel

        stats = BaseEmbeddingModel.cache_stats()
        assert "size" in stats
        assert "max_size" in stats
        assert stats["max_size"] == 10000

    def test_clear_cache(self):
        """Test cache clear method"""
        from src.models.embedding import BaseEmbeddingModel, _embedding_cache

        # Add something to cache
        _embedding_cache["test_key"] = [0.1, 0.2, 0.3]
        assert BaseEmbeddingModel.cache_stats()["size"] >= 1

        # Clear and verify
        BaseEmbeddingModel.clear_cache()
        assert BaseEmbeddingModel.cache_stats()["size"] == 0


class TestLogger:
    """Test unified logging"""

    def test_get_logger_returns_logger(self):
        """get_logger should return a logging.Logger instance"""
        import logging

        from src.utils.logger import get_logger

        logger = get_logger("test_module")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test_module"

    def test_get_logger_same_name_same_instance(self):
        """Same name should return same logger instance"""
        from src.utils.logger import get_logger

        logger1 = get_logger("test_same")
        logger2 = get_logger("test_same")
        assert logger1 is logger2


class TestSelectModel:
    """Test model selection"""

    def test_select_model_import(self):
        """select_model should be importable"""
        from src.models import select_model

        assert callable(select_model)

    @patch("src.models.chat_model.OpenAI")
    def test_select_model_siliconflow(self, mock_openai):
        """select_model should work with siliconflow provider"""
        mock_openai.return_value = MagicMock()

        from src.models import select_model

        # Provide a dummy key so selection does not depend on local `.env`.
        with patch.dict("os.environ", {"SILICONFLOW_API_KEY": "test"}):
            model = select_model(model_provider="siliconflow", model_name="test-model")
            assert model is not None
