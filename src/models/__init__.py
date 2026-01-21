import os
import traceback

from src import config
from src.core.settings import settings
from src.models.chat_model import OpenAIBase
from src.utils.logger import LogManager

logger = LogManager()


def _get_custom_model(custom_id: str) -> dict | None:
    for m in config.get("custom_models", []) or []:
        if m.get("custom_id") == custom_id:
            return m
    return None


def select_model(model_provider: str | None = None, model_name: str | None = None):
    """
    Select a chat model instance.

    - `model_provider`/`model_name` default to values in `config`.
    - Allows server boot without provider API keys so the UI can be used to configure them.
    """

    model_provider = model_provider or config.model_provider
    model_info = config.model_names.get(model_provider, {}) if model_provider else {}
    model_name = model_name or config.model_name or model_info.get("default", "") or settings.llm.model_name

    logger.info(f"Selecting model from `{model_provider}` with `{model_name}`")

    if not model_provider:
        raise ValueError(
            "Model provider not specified. Update `model_provider` in "
            f"`{config.filename}` or set it via the /config API."
        )

    # Custom OpenAI-compatible endpoints configured from UI (/config)
    if model_provider == "custom":
        item = _get_custom_model(model_name)
        if not item:
            raise ValueError(f"Custom model '{model_name}' not found in config.custom_models")

        return OpenAIBase(
            api_key=item.get("api_key") or settings.llm.api_key,
            base_url=item.get("base_url") or item.get("api_base") or settings.llm.api_base,
            model_name=item.get("model_name") or item.get("model") or item.get("custom_id") or model_name,
        )

    if model_provider == "dashscope":
        from src.models.chat_model import DashScope

        return DashScope(model_name)

    if model_provider == "openai":
        from src.models.chat_model import OpenModel

        return OpenModel(model_name)

    # Other OpenAI-compatible providers from models.yaml
    try:
        env_keys = model_info.get("env") or []
        api_key = os.getenv(env_keys[0]) if env_keys else ""
        if not api_key:
            api_key = settings.get_api_key(model_provider) or settings.llm.api_key

        base_url = model_info.get("base_url") or settings.llm.api_base

        if not api_key:
            raise ValueError(
                f"Missing API key for provider '{model_provider}'. "
                f"Set one of: {env_keys or ['llm_api_key']}"
            )

        return OpenAIBase(
            api_key=api_key,
            base_url=base_url,
            model_name=model_name,
        )
    except Exception as e:
        raise ValueError(f"Model provider {model_provider} load failed: {e}\n{traceback.format_exc()}")

