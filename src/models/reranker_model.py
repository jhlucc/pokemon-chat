"""
Reranker models - Remote API only

Supported providers:
- siliconflow: SiliconFlow API (recommended)
- local: HuggingFace model (requires torch, transformers)
"""
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import os
import requests
import numpy as np
import logging
from typing import List, Tuple, Union


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class SiliconFlowReranker:
    """SiliconFlow reranker API (recommended, no torch required)"""
    
    def __init__(self, model_name):
        self.url = "https://api.siliconflow.cn/v1/rerank"
        self.model = model_name
        api_key = os.getenv("SILICONFLOW_API_KEY")
        if not api_key:
            raise ValueError("Please set SILICONFLOW_API_KEY environment variable")
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    def compute_score(self, sentence_pairs: Tuple[str, List[str]], normalize=False):
        query, sentences = sentence_pairs
        payload = {
            "model": self.model,
            "query": query,
            "documents": sentences,
            "max_chunks_per_doc": 512
        }
        response = requests.post(self.url, json=payload, headers=self.headers)
        response_json = response.json()
        if "results" not in response_json:
            raise ValueError(f"Invalid response: {response.text}")

        results = sorted(response_json["results"], key=lambda x: x["index"])
        scores = [r["relevance_score"] for r in results]
        return [sigmoid(s) for s in scores] if normalize else scores


# Optional: Local reranker (requires torch)
HuggingfaceReranker = None

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    
    class HuggingfaceReranker:
        """Local Huggingface reranker (requires torch and transformers)"""
        
        def __init__(self, model_dir, device="cpu"):
            self.device = device
            logging.info(f"Loading Huggingface reranker model from {model_dir} on {device}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True)
            self.model.to(device)
            self.model.eval()
            logging.info("Huggingface model and tokenizer loaded successfully.")

        def compute_score(self, pairs: List[Tuple[str, str]], normalize=True):
            inputs = self.tokenizer(
                [q for q, d in pairs],
                [d for q, d in pairs],
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512,
            ).to(self.device)

            with torch.no_grad():
                logits = self.model(**inputs).logits.squeeze(-1)
                scores = logits.cpu().numpy()

            return sigmoid(scores).tolist() if normalize else scores.tolist()

except ImportError:
    pass  # torch not available, local reranker disabled


class RerankerWrapper:
    """Unified reranker interface"""
    
    def __init__(self, reranker_key, model_name, local_path=None, device="cpu"):
        self.device = device
        self.reranker_key = reranker_key
        provider, short_name = reranker_key.split("/", 1)
        provider = provider.lower()

        if provider == "siliconflow":
            self.reranker = SiliconFlowReranker(model_name)
        elif provider == "local":
            if HuggingfaceReranker is None:
                raise ImportError(
                    "本地 Huggingface reranker 需要安装 torch 和 transformers。\n"
                    "请运行: pip install torch transformers\n"
                    "或者使用远程 API: siliconflow/xxx"
                )
            if not local_path or not os.path.isdir(local_path):
                raise ValueError(f"local_path = {local_path} 不存在!")
            self.reranker = HuggingfaceReranker(local_path, device)
        else:
            raise ValueError(f"Invalid reranker provider: {provider}. Supported: siliconflow, local")

    def run(self, query: str, docs: List[str], normalize=True):
        """Compute rerank scores"""
        if isinstance(self.reranker, SiliconFlowReranker):
            return self.reranker.compute_score((query, docs), normalize=normalize)
        else:
            pairs = [(query, doc) for doc in docs]
            return self.reranker.compute_score(pairs, normalize=normalize)


if __name__ == '__main__':
    query = "皮卡丘的进化是什么？"
    docs = [
        "皮卡丘可以进化为雷丘。",
        "小火龙是初代宝可梦之一。",
        "天气真好，适合去散步。"
    ]

    # Recommended: SiliconFlow API (no torch required)
    reranker = RerankerWrapper("siliconflow/bge-reranker-v2-m3", model_name="BAAI/bge-reranker-v2-m3")

    scores = reranker.run(query, docs)
    for doc, score in zip(docs, scores):
        print(f"{doc}\nScore: {score:.4f}\n")
