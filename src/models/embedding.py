"""
Unified Embedding - 统一的多厂商 Embedding 接口

支持的提供商:
- siliconflow: SiliconFlow API
- openai: OpenAI 兼容 API
- ollama: Ollama 本地服务
- dashscope: 阿里 DashScope API

配置通过 .env 文件或环境变量:
  EMBEDDING_PROVIDER=siliconflow
  EMBEDDING_MODEL=BAAI/bge-m3
"""
import warnings
warnings.filterwarnings("ignore")

import hashlib
import requests
from abc import ABC, abstractmethod
from typing import List, Dict, Union, Any
from src.core.settings import settings
from src.utils import logger

_log = logger.LogManager()


def hashstr(data: Union[str, List[str]]) -> str:
    if isinstance(data, list):
        data = "".join(data)
    return hashlib.md5(data.encode("utf-8")).hexdigest()


class BaseEmbeddingModel(ABC):
    """Embedding 基类"""
    embed_state: Dict[str, Any] = {}
    dimension: int = 1024
    
    @abstractmethod
    def embed(self, texts: Union[str, List[str]]) -> List[List[float]]:
        """生成文本向量"""
        pass
    
    def encode(self, message: Union[str, List[str]]) -> List[List[float]]:
        """兼容旧接口"""
        return self.embed(message)
    
    def predict(self, message: Union[str, List[str]]) -> List[List[float]]:
        """兼容旧接口"""
        return self.embed(message)
    
    def encode_queries(self, queries: Union[str, List[str]]) -> List[List[float]]:
        return self.embed(queries)
    
    def get_dimension(self) -> int:
        return self.dimension
    
    def batch_encode(self, messages: List[str], batch_size: int = 20) -> List[List[float]]:
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
            response = self.embed(group_msg)
            data.extend(response)
            
            if task_id:
                self.embed_state[task_id]['progress'] = i + len(group_msg)

        if task_id:
            self.embed_state[task_id]['progress'] = len(messages)
            self.embed_state[task_id]['status'] = 'completed'

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
        
        _log.info(f"Using SiliconFlow embedding: {self.model}")
    
    def embed(self, texts: Union[str, List[str]]) -> List[List[float]]:
        if isinstance(texts, str):
            texts = [texts]
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "input": texts
        }
        
        response = requests.post(self.url, headers=headers, json=payload)
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
        
        _log.info(f"Using OpenAI-compatible embedding: {self.model} from {self.base_url}")
    
    def embed(self, texts: Union[str, List[str]]) -> List[List[float]]:
        if isinstance(texts, str):
            texts = [texts]
        
        url = f"{self.base_url}/embeddings"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "input": texts
        }
        
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
        
        _log.info(f"Using Ollama embedding: {self.model} at {self.url}")
    
    def embed(self, texts: Union[str, List[str]]) -> List[List[float]]:
        if isinstance(texts, str):
            texts = [texts]
        
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
        
        _log.info(f"Using DashScope embedding: {self.model}")
    
    def embed(self, texts: Union[str, List[str]]) -> List[List[float]]:
        if isinstance(texts, str):
            texts = [texts]
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "input": texts
        }
        
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


def get_embedding_model(
    provider: str = None,
    model: str = None,
    **kwargs
) -> BaseEmbeddingModel:
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
        raise ValueError(
            f"不支持的 Embedding 提供商: {provider}\n"
            f"支持的提供商: {list(EMBEDDING_PROVIDERS.keys())}"
        )
    
    return EMBEDDING_PROVIDERS[provider](model=model, **kwargs)


if __name__ == "__main__":
    print(f"当前 Embedding 配置:")
    print(f"  Provider: {settings.embedding.provider}")
    print(f"  Model: {settings.embedding.model_name}")
    print(f"  Dimension: {settings.embedding.dimension}")
    
    try:
        embedding = get_embedding_model()
        result = embedding.embed("测试文本")
        print(f"\n测试成功! 向量维度: {len(result[0])}")
    except Exception as e:
        print(f"测试失败: {e}")
