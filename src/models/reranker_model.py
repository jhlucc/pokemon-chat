import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import os
import json
import requests
import numpy as np
import logging
import torch
from typing import List, Tuple, Union
from transformers import AutoTokenizer, AutoModelForSequenceClassification

try:
    from FlagEmbedding import FlagReranker  # 可选依赖
except ImportError:
    FlagReranker = None


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class HuggingfaceReranker:
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


class LocalFlagReranker(FlagReranker):
    def __init__(self, model_name_or_path, device="cpu", **kwargs):
        if FlagReranker is None:
            raise ImportError("FlagEmbedding is not installed. Please install it first.")
        super().__init__(model_name_or_path, use_fp16=True, device=device, **kwargs)


class SiliconFlowReranker:
    def __init__(self, model_name):
        self.url = "https://api.siliconflow.cn/v1/rerank"
        self.model = model_name
        self.headers = {
            "Authorization": "Bearer sk-airshplskaflsntrycgajclaomhovoycgmcckhkkqmdvtjfi",
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


class RerankerWrapper:
    def __init__(self, reranker_key, model_name, local_path=None, device="cpu"):
        self.device = device
        self.reranker_key = reranker_key
        provider, short_name = reranker_key.split("/", 1)
        if not os.path.isdir(local_path):
            raise ValueError(f"local_path = {local_path} 不存在!")

        # 自动选择后端并初始化reranker属性
        if provider == "local":
            self.reranker = HuggingfaceReranker(local_path, device)
        elif provider == "Flag":
            self.reranker = LocalFlagReranker(model_name)
        elif provider == "siliconflow":
            self.reranker = SiliconFlowReranker(model_name)
        else:
            raise ValueError(f"Invalid reranker provider: {provider}")

    def run(self, query: str, docs: List[str], normalize=True):
        """
        调用不同后端的 reranker 来计算分数
        """
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

    #  Huggingface
    reranker = RerankerWrapper(
        reranker_key="local/bge-reranker-v2-m3",
        model_name="bge-reranker-v2-m3",
        local_path="C:/Users/luke/Desktop/Smart-Assistant/resources/models/bge-reranker-v2-m3"
    )
    # Flagembedding
    # reranker = RerankerWrapper(
    #     reranker_key="Flag/bge-reranker-v2-m3",
    #     model_name="BAAI/bge-reranker-v2-m3",
    # )

    # SiliconFlowAPI
    # reranker = RerankerWrapper("siliconflow/bge-reranker-v2-m3", model_name="BAAI/bge-reranker-v2-m3")

    scores = reranker.run(query, docs)
    for doc, score in zip(docs, scores):
        print(f"{doc}\nScore: {score:.4f}\n")
