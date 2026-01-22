from typing import Dict, List, Any
import numpy as np

class MockHybridEmbedding:
    def __init__(self, dense_dim: int = 1024):
        self.dim = dense_dim

    def embed_query(self, text: str) -> List[float]:
        return np.random.rand(self.dim).tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [np.random.rand(self.dim).tolist() for _ in texts]
        
    def embed_sparse_query(self, text: str) -> Dict[int, float]:
        # Mock sparse vector (id -> score)
        return {hash(word) % 1000: 0.5 for word in text.split()}

    def embed_sparse_documents(self, texts: List[str]) -> List[Dict[int, float]]:
        return [self.embed_sparse_query(t) for t in texts]
