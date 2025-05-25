import os
import traceback
from src import config
from src.utils.logger import LogManager
from src.models.chat_model import OpenAIBase
logger=LogManager()
from configs.settings import *
from dotenv import load_dotenv
load_dotenv("src/.env")
def select_model(model_provider=None, model_name=None):
    """根据模型提供者选择模型"""
    model_provider = model_provider or config.model_provider
    model_info = config.model_names.get(model_provider, {})
    model_name = model_name or config.model_name or model_info.get("default", "")


    logger.info(f"Selecting model from `{model_provider}` with `{model_name}`")


    if model_provider is None:
        raise ValueError("Model provider not specified, please modify `model_provider` in `src/config/base.yaml`")


    if model_provider == "dashscope":
        from src.models.chat_model import DashScope
        return DashScope(model_name)

    if model_provider == "openai":
        from src.models.chat_model import OpenModel
        return OpenModel(model_name)


    # 其他模型，默认使用OpenAIBase
    try:
        model = OpenAIBase(
            api_key=os.getenv(model_info["env"][0]),
            base_url=model_info["base_url"],
            model_name=model_name,
        )
        return model
    except Exception as e:
        raise ValueError(f"Model provider {model_provider} load failed, {e} \n {traceback.format_exc()}")
