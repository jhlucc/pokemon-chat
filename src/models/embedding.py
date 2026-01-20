#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Embedding models - Remote API only (no local models)

Supported providers:
- openai: OpenAI-compatible API (SiliconFlow, etc.)
- ollama: Local Ollama server
- siliconflow: SiliconFlow API
"""
import warnings

warnings.filterwarnings("ignore")
import os
import hashlib
import requests
from typing import List, Dict, Union, Any

from src.utils import logger
from configs.settings import *
from configs.settings import EMBED_MODEL_INFO

_log = logger.LogManager()


def hashstr(data: Union[str, List[str]]) -> str:
    if isinstance(data, list):
        data = "".join(data)
    return hashlib.md5(data.encode("utf-8")).hexdigest()


class BaseEmbeddingModel:
    embed_state: Dict[str, Any] = {}

    def get_dimension(self) -> Union[int, None]:
        if hasattr(self, "dimension"):
            return self.dimension
        if hasattr(self, "model") and self.model in EMBED_MODEL_INFO:
            return EMBED_MODEL_INFO[self.model].get("dimension")
        return None

    def encode(self, message: Union[str, List[str]]) -> Any:
        return self.predict(message)

    def encode_queries(self, queries: Union[str, List[str]]) -> Any:
        return self.predict(queries)

    def batch_encode(self, messages: List[str], batch_size: int = 20) -> List[Any]:
        _log.info(f"Batch encoding {len(messages)} messages")
        data = []
        task_id = None

        if len(messages) > batch_size:
            task_id = hashstr(messages)
            self.embed_state[task_id] = {
                'status': 'in-progress',
                'total': len(messages),
                'progress': 0
            }
        for i in range(0, len(messages), batch_size):
            group_msg = messages[i: i + batch_size]
            _log.info(f"Encoding messages {i} to {i + batch_size} out of {len(messages)}")
            response = self.encode(group_msg)
            if isinstance(response, list) and len(response) > 0:
                _log.debug(
                    f"Response: len(response)={len(response)}, group_msg count={len(group_msg)}, first emb. length={len(response[0]) if hasattr(response[0], '__len__') else 'N/A'}")
            data.extend(response)

        if task_id:
            self.embed_state[task_id]['progress'] = len(messages)
            self.embed_state[task_id]['status'] = 'completed'

        return data


class OllamaEmbedding(BaseEmbeddingModel):
    """Ollama embedding via local Ollama server"""
    
    def __init__(self, config) -> None:
        info = EMBED_MODEL_INFO[config.embed_model]
        self.model = info["name"]
        self.url = info.get("url", "http://localhost:11434/api/embed")
        self.dimension = info.get("dimension")
        _log.info(f"Using Ollama embedding model `{self.model}` at `{self.url}`")

    def predict(self, message: Union[str, List[str]]) -> List[Any]:
        if isinstance(message, str):
            message = [message]
        payload = {
            "model": self.model,
            "input": message,
        }
        response = requests.post(self.url, json=payload)
        try:
            response_json = response.json()
        except Exception as e:
            raise RuntimeError(f"Failed to decode JSON response: {e}")
        if not response_json.get("embeddings"):
            raise RuntimeError(f"Ollama Embedding failed: {response_json}")
        return response_json["embeddings"]


class OpenAIEmbedding(BaseEmbeddingModel):
    """OpenAI-compatible embedding API (supports SiliconFlow, etc.)"""
    
    def __init__(self, config) -> None:
        self.api_key = os.getenv("OPENAI_API_KEY", MODEL_API_KEY)
        self.base_url = os.getenv("OPENAI_API_BASE", MODEL_API_BASE)
        self.model = EMBEDDING_MODEL
        self.dimension = EMBEDDING_MODEL_DIM
        _log.info(f"Using OpenAI-compatible embedding model `{self.model}` from `{self.base_url}`")

    def predict(self, message: Union[str, List[str]]) -> List[Any]:
        if isinstance(message, str):
            message = [message]
        url = f"{self.base_url}/embeddings"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "input": message
        }
        response = requests.post(url, headers=headers, json=payload)

        if response.status_code != 200:
            raise RuntimeError(
                f"OpenAI embedding request failed with status {response.status_code}. "
                f"Body: {response.text}"
            )
        try:
            result = response.json()
        except Exception as e:
            raise RuntimeError(f"OpenAI embedding response decode error: {e}")
        if "data" not in result:
            raise RuntimeError(f"OpenAI embedding error: {result}")
        embeddings = [d["embedding"] for d in result["data"]]
        if any(len(vec) != self.dimension for vec in embeddings):
            raise RuntimeError(
                f"Received embedding with unexpected dimension. Expecting {self.dimension}."
            )
        return embeddings


class SiliconFlowEmbedding(BaseEmbeddingModel):
    """SiliconFlow embedding API"""
    
    def __init__(self, config) -> None:
        info = EMBED_MODEL_INFO[config.embed_model]
        self.model = info["name"]
        self.dimension = info.get("dimension", 1024)
        self.url = info.get("url", "https://api.siliconflow.cn/v1/embeddings")
        api_key_env = info.get("api_key", "SILICONFLOW_API_KEY")
        self.api_key = os.getenv(api_key_env)
        if not self.api_key:
            raise ValueError(f"Please set {api_key_env} environment variable")
        _log.info(f"Using SiliconFlow embedding model `{self.model}`")

    def predict(self, message: Union[str, List[str]]) -> List[Any]:
        if isinstance(message, str):
            message = [message]
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "input": message
        }
        response = requests.post(self.url, headers=headers, json=payload)

        if response.status_code != 200:
            raise RuntimeError(
                f"SiliconFlow embedding request failed with status {response.status_code}. "
                f"Body: {response.text}"
            )
        try:
            result = response.json()
        except Exception as e:
            raise RuntimeError(f"SiliconFlow embedding response decode error: {e}")
        if "data" not in result:
            raise RuntimeError(f"SiliconFlow embedding error: {result}")
        return [d["embedding"] for d in result["data"]]


def get_embedding_model(config) -> Union[BaseEmbeddingModel, None]:
    """
    Get embedding model based on config.
    
    Supported providers:
    - openai/xxx: OpenAI-compatible API
    - ollama/xxx: Local Ollama server
    - siliconflow/xxx: SiliconFlow API
    
    Note: Local embedding models are no longer supported.
    Please use remote API providers instead.
    """
    if isinstance(config, dict):
        class ConfigObject:
            def __init__(self, config_dict):
                for k, v in config_dict.items():
                    setattr(self, k, v)
        config = ConfigObject(config)
    
    if not getattr(config, "enable_knowledge_base", False):
        return None
    
    _log.debug(f"Loading embedding model: {config.embed_model}")
    provider, _ = config.embed_model.split('/', 1)
    provider = provider.lower()
    
    if provider == "local":
        raise ValueError(
            "本地 embedding 已不再支持。请使用远程 API 服务：\n"
            "- openai/xxx: OpenAI 兼容 API\n"
            "- ollama/xxx: Ollama 本地服务\n"
            "- siliconflow/xxx: SiliconFlow API"
        )
    elif provider == "ollama":
        return OllamaEmbedding(config)
    elif provider == "openai":
        return OpenAIEmbedding(config)
    elif provider == "siliconflow":
        return SiliconFlowEmbedding(config)
    else:
        raise ValueError(f"Unsupported embedding provider: {provider}")


if __name__ == "__main__":
    class Config:
        enable_knowledge_base = True
        # Use SiliconFlow API for embedding
        embed_model = "siliconflow/BAAI/bge-m3"

    config = Config()
    embedding_model = get_embedding_model(config)

    if embedding_model:
        single_message = "请简单介绍一下人工智能的发展历程。"
        try:
            single_result = embedding_model.encode(single_message)
            _log.info(f"单条编码结果维度: {len(single_result[0])}")
        except Exception as e:
            _log.error(f"编码调用失败: {e}")
    else:
        _log.error("知识库功能未启用。")
