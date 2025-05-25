#!/usr/bin/env python
# -*- coding: utf-8 -*-
import warnings

warnings.filterwarnings("ignore")
import os
import json
import hashlib
import requests
from typing import List, Dict, Union, Generator, Any

from FlagEmbedding import FlagModel
from src.utils import logger
from configs.settings import *
from configs.settings import EMBED_MODEL_INFO, MODEL_EMBEDDING_PATH

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


class LocalEmbeddingModel(FlagModel, BaseEmbeddingModel):
    def __init__(self, config, **kwargs):
        info = EMBED_MODEL_INFO[config.embed_model]
        self.model = info["name"]
        self.dimension = info["dimension"]

        path_from_env = os.getenv("MODEL_EMBEDDING_PATH")
        if path_from_env and os.path.exists(path_from_env):
            resolved_path = path_from_env
        elif MODEL_EMBEDDING_PATH and os.path.exists(MODEL_EMBEDDING_PATH):
            resolved_path = MODEL_EMBEDDING_PATH
        elif info.get("local_path") and os.path.exists(info["local_path"]):
            resolved_path = info["local_path"]
        else:
            raise FileNotFoundError("无法定位本地嵌入模型路径，请检查MODEL_EMBEDDING_PATH 或 local_path 配置")

        self.model = resolved_path
        super().__init__(
            self.model,
            query_instruction_for_retrieval=info.get("query_instruction", ""),
            use_fp16=False,
            **kwargs
        )
        _log.info(f"Embedding model `{info['name']}` loaded.")


class OllamaEmbedding(BaseEmbeddingModel):
    def __init__(self, config) -> None:
        info = EMBED_MODEL_INFO[
            config.embed_model]  # {'name': 'bge-m3:latest  ', 'dimension': 1024, 'url': 'http://localhost:11434/api/embed'}
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
            # OpenAIEmbedding 类中 predict() 方法里，加上打印
            # print("实际返回维度:", len(result["data"][0]["embedding"]))

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


class OtherEmbedding(BaseEmbeddingModel):
    def __init__(self, config) -> None:
        raise NotImplementedError("OtherEmbedding is not implemented yet.")


def get_embedding_model(config) -> Union[BaseEmbeddingModel, None]:
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
    if provider.lower() == "local":
        return LocalEmbeddingModel(config)
    elif provider.lower() == "ollama":
        return OllamaEmbedding(config)
    elif provider.lower() == "openai":
        return OpenAIEmbedding(config)
    else:
        return OtherEmbedding(config)


if __name__ == "__main__":
    class Config:
        # 启用知识库功能
        enable_knowledge_base = True
        # 指定 embedding 模型，格式为 "provider/bge-large-zh-v1.5"，此处选择本地模型
        embed_model = "local/bge-large-zh-v1.5"
        # embed_model = "ollama/bge-m3:latest"
        # embed_model = "openai/bge-m3-pro"


    # 创建配置实例
    config = Config()
    # 调用辅助函数加载embedding模型
    embedding_model = get_embedding_model(config)

    if embedding_model:
        # 对单条输入文本进行编码调用
        single_message = "请简单介绍一下人工智能的发展历程。"
        try:
            single_result = embedding_model.encode(single_message)
            _log.info(f"单条编码结果: {single_result}")
        except Exception as e:
            _log.error(f"单条编码调用失败: {e}")

        # 演示批量调用
        messages = [
            "人工智能是什么？",
            "机器学习与人工智能的关系是什么？",
            "请介绍一下深度学习的基本原理。"
        ]
        try:
            batch_result = embedding_model.batch_encode(messages, batch_size=2)
            _log.info(f"批量编码结果: {batch_result}")
        except Exception as e:
            _log.error(f"批量编码调用失败: {e}")
    else:
        _log.error("知识库功能未启用或无法加载 embedding 模型。")
