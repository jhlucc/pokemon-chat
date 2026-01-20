"""
Unified Reranker - 统一的多厂商 Reranker 接口

支持的提供商:
- siliconflow: SiliconFlow API
- dashscope: 阿里 DashScope API
- jina: Jina AI API
- cohere: Cohere API

配置通过 .env 文件或环境变量:
  RERANKER_PROVIDER=siliconflow
  RERANKER_MODEL=BAAI/bge-reranker-v2-m3
"""
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import requests
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple, Union
from src.core.settings import settings


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class BaseReranker(ABC):
    """Reranker 基类"""
    
    @abstractmethod
    def rerank(self, query: str, documents: List[str], top_k: int = 10) -> List[Tuple[int, float, str]]:
        """
        对文档进行重排序
        
        Args:
            query: 查询文本
            documents: 待排序的文档列表
            top_k: 返回的最大结果数
            
        Returns:
            List of (index, score, document) 按相关性降序排列
        """
        pass
    
    def compute_score(self, query: str, documents: List[str], normalize: bool = False) -> List[float]:
        """计算相关性分数"""
        results = self.rerank(query, documents, top_k=len(documents))
        # 按原始索引排序返回分数
        scores = [0.0] * len(documents)
        for idx, score, _ in results:
            scores[idx] = sigmoid(score) if normalize else score
        return scores


class SiliconFlowReranker(BaseReranker):
    """SiliconFlow Reranker API"""
    
    def __init__(self, model: str = None, api_key: str = None):
        self.model = model or settings.reranker.model
        self.api_key = api_key or settings.get_api_key("siliconflow")
        self.url = "https://api.siliconflow.cn/v1/rerank"
        
        if not self.api_key:
            raise ValueError("请设置 SILICONFLOW_API_KEY 环境变量")
    
    def rerank(self, query: str, documents: List[str], top_k: int = 10) -> List[Tuple[int, float, str]]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "query": query,
            "documents": documents,
            "top_n": top_k,
        }
        
        response = requests.post(self.url, json=payload, headers=headers)
        if response.status_code != 200:
            raise RuntimeError(f"SiliconFlow rerank failed: {response.text}")
        
        result = response.json()
        if "results" not in result:
            raise RuntimeError(f"Invalid response: {result}")
        
        return [
            (r["index"], r["relevance_score"], documents[r["index"]])
            for r in result["results"]
        ]


class DashScopeReranker(BaseReranker):
    """阿里 DashScope Reranker API"""
    
    def __init__(self, model: str = None, api_key: str = None):
        self.model = model or settings.reranker.model or "gte-rerank"
        self.api_key = api_key or settings.get_api_key("dashscope")
        self.url = "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/v1/rerank"
        
        if not self.api_key:
            raise ValueError("请设置 DASHSCOPE_API_KEY 环境变量")
    
    def rerank(self, query: str, documents: List[str], top_k: int = 10) -> List[Tuple[int, float, str]]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "input": {
                "query": query,
                "documents": documents
            },
            "parameters": {
                "top_n": top_k,
                "return_documents": False
            }
        }
        
        response = requests.post(self.url, json=payload, headers=headers)
        if response.status_code != 200:
            raise RuntimeError(f"DashScope rerank failed: {response.text}")
        
        result = response.json()
        if "output" not in result or "results" not in result["output"]:
            raise RuntimeError(f"Invalid response: {result}")
        
        return [
            (r["index"], r["relevance_score"], documents[r["index"]])
            for r in result["output"]["results"]
        ]


class JinaReranker(BaseReranker):
    """Jina AI Reranker API"""
    
    def __init__(self, model: str = None, api_key: str = None):
        self.model = model or settings.reranker.model or "jina-reranker-v2-base-multilingual"
        self.api_key = api_key or settings.get_api_key("jina")
        self.url = "https://api.jina.ai/v1/rerank"
        
        if not self.api_key:
            raise ValueError("请设置 JINA_API_KEY 环境变量")
    
    def rerank(self, query: str, documents: List[str], top_k: int = 10) -> List[Tuple[int, float, str]]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "query": query,
            "documents": documents,
            "top_n": top_k
        }
        
        response = requests.post(self.url, json=payload, headers=headers)
        if response.status_code != 200:
            raise RuntimeError(f"Jina rerank failed: {response.text}")
        
        result = response.json()
        if "results" not in result:
            raise RuntimeError(f"Invalid response: {result}")
        
        return [
            (r["index"], r["relevance_score"], documents[r["index"]])
            for r in result["results"]
        ]


class CohereReranker(BaseReranker):
    """Cohere Reranker API"""
    
    def __init__(self, model: str = None, api_key: str = None):
        self.model = model or settings.reranker.model or "rerank-multilingual-v3.0"
        self.api_key = api_key or settings.get_api_key("cohere")
        self.url = "https://api.cohere.ai/v1/rerank"
        
        if not self.api_key:
            raise ValueError("请设置 COHERE_API_KEY 环境变量")
    
    def rerank(self, query: str, documents: List[str], top_k: int = 10) -> List[Tuple[int, float, str]]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "query": query,
            "documents": documents,
            "top_n": top_k,
            "return_documents": False
        }
        
        response = requests.post(self.url, json=payload, headers=headers)
        if response.status_code != 200:
            raise RuntimeError(f"Cohere rerank failed: {response.text}")
        
        result = response.json()
        if "results" not in result:
            raise RuntimeError(f"Invalid response: {result}")
        
        return [
            (r["index"], r["relevance_score"], documents[r["index"]])
            for r in result["results"]
        ]


# Reranker 工厂
RERANKER_PROVIDERS = {
    "siliconflow": SiliconFlowReranker,
    "dashscope": DashScopeReranker,
    "jina": JinaReranker,
    "cohere": CohereReranker,
}


def get_reranker(
    provider: str = None,
    model: str = None,
    api_key: str = None
) -> BaseReranker:
    """
    获取 Reranker 实例
    
    Args:
        provider: 提供商名称，默认从 settings 读取
        model: 模型名称，默认从 settings 读取
        api_key: API Key，默认从 settings 读取
        
    Returns:
        BaseReranker 实例
    """
    provider = provider or settings.reranker.provider
    provider = provider.lower()
    
    if provider not in RERANKER_PROVIDERS:
        raise ValueError(
            f"不支持的 Reranker 提供商: {provider}\n"
            f"支持的提供商: {list(RERANKER_PROVIDERS.keys())}"
        )
    
    return RERANKER_PROVIDERS[provider](model=model, api_key=api_key)


# 兼容旧接口
class RerankerWrapper:
    """兼容旧版接口的包装类"""
    
    def __init__(self, reranker_key: str = None, model_name: str = None, **kwargs):
        if reranker_key:
            provider, _ = reranker_key.split("/", 1)
        else:
            provider = settings.reranker.provider
        self.reranker = get_reranker(provider=provider, model=model_name)
    
    def run(self, query: str, docs: List[str], normalize: bool = True) -> List[float]:
        return self.reranker.compute_score(query, docs, normalize=normalize)


if __name__ == '__main__':
    # 测试
    print(f"当前 Reranker 配置:")
    print(f"  Provider: {settings.reranker.provider}")
    print(f"  Model: {settings.reranker.model}")
    
    query = "皮卡丘的进化是什么？"
    docs = [
        "皮卡丘可以进化为雷丘。",
        "小火龙是初代宝可梦之一。",
        "天气真好，适合去散步。"
    ]
    
    try:
        reranker = get_reranker()
        results = reranker.rerank(query, docs)
        print(f"\n查询: {query}")
        print("排序结果:")
        for idx, score, doc in results:
            print(f"  [{idx}] {score:.4f} - {doc}")
    except Exception as e:
        print(f"测试失败: {e}")
