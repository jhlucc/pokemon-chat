#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import logging
import traceback
from typing import Any, Dict, List, Callable

from pymilvus import connections, Collection
from src.core.settings import settings
from src.models.embedding import get_embedding_model

try:
    from src.models.reranker_model import RerankerWrapper

    RERANK_AVAILABLE = True
except ImportError:
    RERANK_AVAILABLE = False

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class VectorRecaller:
    def __init__(self) -> None:
        # 解析 milvus_uri 获取 host 和 port
        milvus_uri = settings.kb_config.milvus_uri
        if milvus_uri.startswith("http://"):
            uri_parts = milvus_uri.replace("http://", "").split(":")
            host = uri_parts[0]
            port = uri_parts[1] if len(uri_parts) > 1 else "19530"
        else:
            host = "localhost"
            port = "19530"
        self.collection_name = "documents_collection1"

        connections.connect(host=host, port=port)
        self.collection = Collection(self.collection_name)
        self.collection.load()

        self.distance_threshold: float = settings.kb_config.default_distance_threshold
        self.rerank_threshold: float = settings.kb_config.default_rerank_threshold
        self.max_query_count: int = settings.kb_config.default_max_query_count
        self.top_k: int = settings.kb_config.default_top_k

        self.embed_model = get_embedding_model(settings)
        if self.embed_model is None:
            logger.error("Embedding model is not loaded.")
            raise ValueError("Embedding model is None")

        self.reranker = None
        if settings.features.enable_reranker and RERANK_AVAILABLE:
            reranker_key = settings.kb_config.reranker_key
            model_name = settings.reranker.model_name
            local_path = str(settings.paths.model_reranker_path)
            self.reranker = RerankerWrapper(reranker_key, model_name, local_path=local_path, device="cpu")
            logger.info("Reranker loaded.")
        else:
            logger.info("Reranker not enabled or not available.")

    def search_by_vector(self, vector: List[float], limit: int = 10) -> List[Any]:
        search_params = {
            "metric_type": "COSINE",
            "params": {"nprobe": 8, "ef": 64}
        }

        results = self.collection.search(
            data=[vector],
            anns_field="embedding",
            param=search_params,
            limit=limit,
            expr="text_length > 50",
            output_fields=["text", "metadata"]
        )

        hits = results[0] if results else []
        return hits

    def search(self, query: str, limit: int = 10) -> List[Any]:
        vectors = self.embed_model.batch_encode([query])
        if not vectors:
            logger.error("Embedding failed.")
            return []
        return self.search_by_vector(vectors[0], limit)

    def query(self, query: str, **kwargs) -> Dict[str, Any]:
        distance_threshold = kwargs.get("distance_threshold", self.distance_threshold)
        rerank_threshold = kwargs.get("rerank_threshold", self.rerank_threshold)
        max_query_count = kwargs.get("max_query_count", self.max_query_count)
        top_k = kwargs.get("top_k", self.top_k)

        hits = self.search(query, limit=max_query_count)

        filtered_hits = [hit for hit in hits if hit.distance < distance_threshold]

        results_with_scores = []
        if self.reranker and filtered_hits:
            texts = [getattr(hit.entity, "text", "") for hit in filtered_hits]
            rerank_scores = self.reranker.run(query, texts, normalize=False)

            for hit, score in zip(filtered_hits, rerank_scores):
                result = {
                    "text": getattr(hit.entity, "text", ""),
                    "metadata": getattr(hit.entity, "metadata", {}),
                    "distance": hit.distance,
                    "rerank_score": score
                }
                results_with_scores.append(result)

            results_with_scores = [
                res for res in results_with_scores if res["rerank_score"] > rerank_threshold
            ]

            results_with_scores.sort(key=lambda x: x["rerank_score"], reverse=True)
        else:
            results_with_scores = [
                {
                    "text": getattr(hit.entity, "text", ""),
                    "metadata": getattr(hit.entity, "metadata", {}),
                    "distance": hit.distance,
                    "rerank_score": None
                }
                for hit in filtered_hits
            ]

        final_results = results_with_scores[:top_k]

        return {
            "results": final_results,
            "all_hits": hits
        }

    def get_retriever(self) -> Callable[[str], List[Dict[str, Any]]]:
        def retriever(query: str) -> List[Dict[str, Any]]:
            return self.query(query).get("results", [])

        return retriever

    def close(self) -> None:
        connections.disconnect("default")
        logger.info("Milvus connection closed.")


# ============ 使用示例 ============
if __name__ == "__main__":
    try:
        recall_agent = VectorRecaller()
        query_text = "Chjfby邮箱地址是多少？"
        results = recall_agent.query(query_text)
        logger.info("Final retrieved results:")
        for res in results.get("results", []):
            logger.info(f"内容: {res['text']}")
            logger.info(f"元数据: {res['metadata']}")
            logger.info(f"相似度: {1 - res['distance']:.3f}")
            if res.get('rerank_score'):
                logger.info(f"重排序分数: {res['rerank_score']:.3f}")
            print("-----")
    except Exception as e:
        logger.error(f"Exception: {e}\n{traceback.format_exc()}")
    finally:
        recall_agent.close()
