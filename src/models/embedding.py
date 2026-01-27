"""
Unified Embedding - 统一的多厂商 Embedding 接口

支持的提供商:
- siliconflow: SiliconFlow API
- openai: OpenAI 兼容 API
- ollama: Ollama 本地服务
- dashscope: 阿里 DashScope API

特性:
- 内置 LRU 缓存，避免重复计算
- 批量编码支持
- 异步 API (aembed, abatch_encode)
"""

import asyncio
import hashlib
import warnings
from abc import ABC, abstractmethod
from typing import Any

import httpx
import requests

from src.core.settings import settings
from src.utils.logger import get_logger

warnings.filterwarnings("ignore")

logger = get_logger(__name__)

# 全局 embedding 缓存 (可配置大小)
_CACHE_SIZE = 10000
_embedding_cache: dict[str, list[float]] = {}


def _cache_key(text: str, model: str) -> str:
    """生成缓存键"""
    return hashlib.md5(f"{model}:{text}".encode()).hexdigest()


def hashstr(data: str | list[str]) -> str:
    if isinstance(data, list):
        data = "".join(data)
    return hashlib.md5(data.encode("utf-8")).hexdigest()


class BaseEmbeddingModel(ABC):
    """Embedding 基类，带缓存支持"""

    embed_state: dict[str, Any] = {}
    dimension: int = 1024
    use_cache: bool = True

    @abstractmethod
    def _embed_impl(self, texts: list[str]) -> list[list[float]]:
        """实际的 embedding 实现 (子类实现)"""
        pass

    def embed(self, texts: str | list[str]) -> list[list[float]]:
        """生成文本向量 (带缓存)"""
        if isinstance(texts, str):
            texts = [texts]

        if not self.use_cache:
            return self._embed_impl(texts)

        # 检查缓存
        results = [None] * len(texts)
        uncached_indices = []
        uncached_texts = []

        for i, text in enumerate(texts):
            key = _cache_key(text, getattr(self, "model", "default"))
            if key in _embedding_cache:
                results[i] = _embedding_cache[key]
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)

        # 仅计算未缓存的
        if uncached_texts:
            logger.debug(f"Cache miss for {len(uncached_texts)}/{len(texts)} texts")
            new_embeddings = self._embed_impl(uncached_texts)

            # 存入缓存
            for idx, text, emb in zip(uncached_indices, uncached_texts, new_embeddings):
                key = _cache_key(text, getattr(self, "model", "default"))
                if len(_embedding_cache) < _CACHE_SIZE:
                    _embedding_cache[key] = emb
                results[idx] = emb
        else:
            logger.debug(f"Cache hit for all {len(texts)} texts")

        return results

    def get_dimension(self) -> int:
        return self.dimension

    def batch_encode(self, messages: list[str], batch_size: int = 20) -> list[list[float]]:
        logger.info(f"Batch encoding {len(messages)} messages")
        data = []
        task_id = None

        if len(messages) > batch_size:
            task_id = hashstr(messages)
            self.embed_state[task_id] = {"status": "in-progress", "total": len(messages), "progress": 0}

        for i in range(0, len(messages), batch_size):
            group_msg = messages[i : i + batch_size]
            logger.info(f"Encoding messages {i} to {i + batch_size} out of {len(messages)}")
            response = self.embed(group_msg)
            data.extend(response)

            if task_id:
                self.embed_state[task_id]["progress"] = i + len(group_msg)

        if task_id:
            self.embed_state[task_id]["progress"] = len(messages)
            self.embed_state[task_id]["status"] = "completed"

        return data

    @classmethod
    def clear_cache(cls):
        """清除 embedding 缓存"""
        global _embedding_cache
        _embedding_cache.clear()
        logger.info("Embedding cache cleared")

    @classmethod
    def cache_stats(cls) -> dict[str, int]:
        """获取缓存统计"""
        return {"size": len(_embedding_cache), "max_size": _CACHE_SIZE}

    # ========== Async Methods ==========

    async def _aembed_impl(self, texts: list[str]) -> list[list[float]]:
        """异步 embedding 实现 (子类可覆盖)

        默认在线程池中运行同步方法
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._embed_impl, texts)

    async def aembed(self, texts: str | list[str]) -> list[list[float]]:
        """异步生成文本向量 (带缓存)"""
        if isinstance(texts, str):
            texts = [texts]

        if not self.use_cache:
            return await self._aembed_impl(texts)

        # 检查缓存
        results = [None] * len(texts)
        uncached_indices = []
        uncached_texts = []

        for i, text in enumerate(texts):
            key = _cache_key(text, getattr(self, "model", "default"))
            if key in _embedding_cache:
                results[i] = _embedding_cache[key]
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)

        # 仅计算未缓存的
        if uncached_texts:
            logger.debug(f"[Async] Cache miss for {len(uncached_texts)}/{len(texts)} texts")
            new_embeddings = await self._aembed_impl(uncached_texts)

            for idx, text, emb in zip(uncached_indices, uncached_texts, new_embeddings):
                key = _cache_key(text, getattr(self, "model", "default"))
                if len(_embedding_cache) < _CACHE_SIZE:
                    _embedding_cache[key] = emb
                results[idx] = emb
        else:
            logger.debug(f"[Async] Cache hit for all {len(texts)} texts")

        return results

    async def abatch_encode(self, messages: list[str], batch_size: int = 20) -> list[list[float]]:
        """异步批量编码"""
        logger.info(f"[Async] Batch encoding {len(messages)} messages")
        data = []
        task_id = None

        if len(messages) > batch_size:
            task_id = hashstr(messages)
            self.embed_state[task_id] = {"status": "in-progress", "total": len(messages), "progress": 0}

        for i in range(0, len(messages), batch_size):
            group_msg = messages[i : i + batch_size]
            logger.info(f"[Async] Encoding messages {i} to {i + batch_size} out of {len(messages)}")
            response = await self.aembed(group_msg)
            data.extend(response)

            if task_id:
                self.embed_state[task_id]["progress"] = i + len(group_msg)

        if task_id:
            self.embed_state[task_id]["progress"] = len(messages)
            self.embed_state[task_id]["status"] = "completed"

        return data


class SiliconFlowEmbedding(BaseEmbeddingModel):
    """SiliconFlow Embedding API"""

    def __init__(self, model: str = None, api_key: str = None, dimension: int = None):
        self.model = model or settings.embedding.model_name
        self.api_key = api_key or settings.embedding.api_key
        self.dimension = dimension or settings.embedding.dimension
        self.url = settings.embedding.api_base

        if not self.api_key:
            raise ValueError("请设置 EMBEDDING_API_KEY 环境变量")

        logger.info(f"Using SiliconFlow embedding: {self.model}")

    def _embed_impl(self, texts: list[str]) -> list[list[float]]:
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {"model": self.model, "input": texts}

        response = requests.post(self.url, headers=headers, json=payload)
        if response.status_code != 200:
            raise RuntimeError(f"SiliconFlow embedding failed: {response.text}")

        result = response.json()
        if "data" not in result:
            raise RuntimeError(f"Invalid response: {result}")

        return [d["embedding"] for d in result["data"]]

    async def _aembed_impl(self, texts: list[str]) -> list[list[float]]:
        """Native async implementation using httpx"""
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {"model": self.model, "input": texts}

        async with httpx.AsyncClient() as client:
            response = await client.post(self.url, headers=headers, json=payload, timeout=30.0)
            if response.status_code != 200:
                raise RuntimeError(f"SiliconFlow embedding failed: {response.text}")

            result = response.json()
            if "data" not in result:
                raise RuntimeError(f"Invalid response: {result}")

            return [d["embedding"] for d in result["data"]]


class OpenAIEmbedding(BaseEmbeddingModel):
    """OpenAI 兼容 Embedding API"""

    def __init__(self, model: str = None, api_key: str = None, base_url: str = None, dimension: int = None):
        self.model = model or settings.embedding.model_name
        self.api_key = api_key or settings.embedding.api_key
        self.base_url = base_url or settings.embedding.api_base
        self.dimension = dimension or settings.embedding.dimension

        if not self.api_key:
            raise ValueError("请设置 EMBEDDING_API_KEY 环境变量")

        logger.info(f"Using OpenAI-compatible embedding: {self.model} from {self.base_url}")

    def _embed_impl(self, texts: list[str]) -> list[list[float]]:
        url = f"{self.base_url}/embeddings"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {"model": self.model, "input": texts}

        response = requests.post(url, headers=headers, json=payload)
        if response.status_code != 200:
            raise RuntimeError(f"OpenAI embedding failed: {response.text}")

        result = response.json()
        if "data" not in result:
            raise RuntimeError(f"Invalid response: {result}")

        return [d["embedding"] for d in result["data"]]


class OllamaEmbedding(BaseEmbeddingModel):
    """Ollama Embedding"""

    def __init__(self, model: str = None, url: str = None, dimension: int = None):
        self.model = model or settings.embedding.model_name
        self.url = url or "http://localhost:11434/api/embeddings"
        self.dimension = dimension or settings.embedding.dimension

        logger.info(f"Using Ollama embedding: {self.model} at {self.url}")

    def _embed_impl(self, texts: list[str]) -> list[list[float]]:
        payload = {
            "model": self.model,
            "input": texts,
        }

        response = requests.post(self.url, json=payload)
        if response.status_code != 200:
            raise RuntimeError(f"Ollama embedding failed: {response.text}")

        result = response.json()
        if not result.get("embeddings"):
            raise RuntimeError(f"Invalid response: {result}")

        return result["embeddings"]


class DashScopeEmbedding(BaseEmbeddingModel):
    """阿里 DashScope Embedding API"""

    def __init__(self, model: str = None, api_key: str = None, dimension: int = None):
        self.model = model or settings.embedding.model_name or "text-embedding-v3"
        self.api_key = api_key or settings.embedding.api_key
        self.dimension = dimension or settings.embedding.dimension
        self.url = "https://dashscope.aliyuncs.com/api/v1/services/embeddings"

        if not self.api_key:
            raise ValueError("请设置 EMBEDDING_API_KEY 环境变量")

        logger.info(f"Using DashScope embedding: {self.model}")

    def _embed_impl(self, texts: list[str]) -> list[list[float]]:
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {"model": self.model, "input": texts}

        response = requests.post(self.url, headers=headers, json=payload)
        if response.status_code != 200:
            raise RuntimeError(f"DashScope embedding failed: {response.text}")

        result = response.json()
        if "data" not in result:
            raise RuntimeError(f"Invalid response: {result}")

        return [d["embedding"] for d in result["data"]]


# Embedding 工厂
EMBEDDING_PROVIDERS = {
    "siliconflow": SiliconFlowEmbedding,
    "openai": OpenAIEmbedding,
    "ollama": OllamaEmbedding,
    "dashscope": DashScopeEmbedding,
}


def get_embedding_model(provider: str = None, model: str = None, **kwargs) -> BaseEmbeddingModel:
    """
    获取 Embedding 模型实例

    Args:
        provider: 提供商名称，默认从 settings 读取
        model: 模型名称，默认从 settings 读取

    Returns:
        BaseEmbeddingModel 实例
    """
    provider = provider or settings.embedding.provider
    provider = provider.lower()

    if provider not in EMBEDDING_PROVIDERS:
        raise ValueError(f"不支持的 Embedding 提供商: {provider}\n支持的提供商: {list(EMBEDDING_PROVIDERS.keys())}")

    return EMBEDDING_PROVIDERS[provider](model=model, **kwargs)


if __name__ == "__main__":
    print("当前 Embedding 配置:")
    print(f"  Provider: {settings.embedding.provider}")
    print(f"  Model: {settings.embedding.model_name}")
    print(f"  Dimension: {settings.embedding.dimension}")

    try:
        embedding = get_embedding_model()
        result = embedding.embed("测试文本")
        print(f"\n测试成功! 向量维度: {len(result[0])}")
    except Exception as e:
        print(f"测试失败: {e}")
