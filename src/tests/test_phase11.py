"""
Tests for Phase 11: Agentic Memory
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestAgenticMemory:
    """Test Agentic Memory functionality."""

    def test_memory_initialization(self):
        with patch("src.knowledge.memory.agentic.settings") as mock_settings:
            mock_settings.paths.data_dir = Path(tempfile.mkdtemp())
            mock_settings.llm.model_name = "test"
            mock_settings.llm.api_key = "test"
            mock_settings.llm.api_base = "http://test"

            from src.knowledge.memory.agentic import AgenticMemory

            # Mock the LLM
            with patch.object(AgenticMemory, "__init__", lambda self, db_path=None: None):
                memory = AgenticMemory.__new__(AgenticMemory)
                memory.db_path = mock_settings.paths.data_dir / "test.db"
                memory._init_db = MagicMock()

                # Test preferences model
                from src.knowledge.memory.agentic import UserPreferences

                prefs = UserPreferences()

                assert prefs.response_style == "balanced"
                assert prefs.favorite_pokemon == []
                assert prefs.interests == []

    def test_user_preferences_model(self):
        from src.knowledge.memory.agentic import UserPreferences

        prefs = UserPreferences(
            favorite_pokemon=["Pikachu", "Charizard"],
            favorite_types=["Fire", "Electric"],
            response_style="brief",
            interests=["battles", "evolution"],
            notes="Loves competitive battling",
        )

        assert "Pikachu" in prefs.favorite_pokemon
        assert prefs.response_style == "brief"

        # Test serialization
        json_str = prefs.model_dump_json()
        assert "Pikachu" in json_str

    def test_system_prompt_injection_format(self):
        from src.knowledge.memory.agentic import AgenticMemory, UserPreferences

        with patch.object(AgenticMemory, "__init__", lambda self, db_path=None: None):
            memory = AgenticMemory.__new__(AgenticMemory)
            memory.db_path = MagicMock()

            # Mock get_preferences
            def mock_get_prefs(user_id):
                return UserPreferences(favorite_pokemon=["Pikachu"], response_style="brief")

            memory.get_preferences = mock_get_prefs

            injection = memory.get_system_prompt_injection("test_user")

            assert "[User Preferences]" in injection
            assert "Pikachu" in injection
            assert "brief" in injection


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
