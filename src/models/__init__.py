"""
Model selection utilities.

This module provides a unified interface for selecting chat models from various providers.
All configuration is read from `src.core.settings.settings`.
"""

import traceback

from src.core.provider_config import get_provider_api_base, get_provider_api_key
from src.core.settings import settings
from src.models.chat_model import OpenAIBase
from src.utils.logger import get_logger

logger = get_logger(__name__)


def select_model(model_provider: str | None = None, model_name: str | None = None):
    """
    Select a chat model instance.

    Args:
        model_provider: Provider name (e.g., 'openai', 'dashscope', 'siliconflow').
                       Defaults to 'siliconflow' if not specified.
        model_name: Model name. Defaults to settings.llm.model_name.

    Returns:
        A chat model instance with `.predict()` method.
    """
    # Default to settings values
    model_provider = model_provider or "siliconflow"
    model_name = model_name or settings.llm.model_name

    logger.info(f"Selecting model from `{model_provider}` with `{model_name}`")

    if model_provider == "dashscope":
        from src.models.chat_model import Bailian

        return Bailian(model_name)

    # Generic OpenAI-compatible providers (siliconflow, etc.)
    try:
        # Prefer UI-configured provider credentials (persisted under resources/save/config),
        # then fall back to env-based compatibility keys, finally llm_*.
        api_key = get_provider_api_key(model_provider) or settings.get_api_key(model_provider) or settings.llm.api_key
        base_url = get_provider_api_base(model_provider) or settings.llm.api_base

        if not api_key:
            raise ValueError(
                f"Missing API key for provider '{model_provider}'. "
                f"Set llm_api_key / provider API key in the Settings UI, or via environment variables."
            )

        return OpenAIBase(
            api_key=api_key,
            base_url=base_url,
            model_name=model_name,
        )
    except Exception as e:
        raise ValueError(f"Model provider {model_provider} load failed: {e}\n{traceback.format_exc()}") from e
