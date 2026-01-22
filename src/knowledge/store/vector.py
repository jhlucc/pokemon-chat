import warnings
import numpy as np
import torch
from typing import List, Optional, Dict, Any, Union
from langchain_core.documents import Document
from pymilvus import (
    connections, FieldSchema, CollectionSchema,
    DataType, Collection, utility, MilvusClient, MilvusException
)
from src.core.settings import settings
from src.utils.logger import get_logger

# Optional dependencies
try:
    from langchain_openai import OpenAIEmbeddings
except ImportError:
    OpenAIEmbeddings = None

try:
    from src.models.reranker_model import RerankerWrapper
except ImportError:
    RerankerWrapper = None

warnings.filterwarnings("ignore", category=FutureWarning)
_log = get_logger(__name__)

class VectorStore:
    """
    Unified Vector Store for Knowledge Base (Milvus).
    Handles connection, schema management, insertion, and retrieval.
    """
    def __init__(
            self,
            collection_name: str = "default",
            dim: int = 1024,
            host: str = "localhost",
            port: str = "19530",
            overwrite: bool = False,
            embedding_model: Optional[Any] = None, # Can be instance or name
            reranker_model: Optional[Any] = None, # Can be instance or name
            connection_alias: str = "default"
    ):
        self.collection_name = collection_name
        self.dim = dim
        self.host = host
        self.port = port
        self.connection_alias = connection_alias
        
        # Initialize models (optional, for on-the-fly embedding/reranking)
        self.embedder = self._resolve_embedder(embedding_model)
        self.reranker = self._resolve_reranker(reranker_model)

        # Connect to Milvus
        self._connect()
        
        # Schemas
        self.fields = [
            FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
            FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
            FieldSchema(name="metadata", dtype=DataType.JSON),
            FieldSchema(name="text_length", dtype=DataType.INT64),
            # Optional: Add file_id explicitly if needed for efficient filtering, 
            # but metadata["file_id"] usually suffices for JSON filter.
        ]
        
        self.schema = CollectionSchema(
            fields=self.fields,
            description="Knowledge Base Documents",
            enable_dynamic_field=True
        )

        # Collection initialization
        self.collection = self._init_collection(overwrite)
        
        # Indexing
        if not self.collection.has_index():
            self._create_index()

    def _connect(self):
        try:
            # Check if connected
            if not connections.has_connection(self.connection_alias):
                _log.info(f"Connecting to Milvus at {self.host}:{self.port}")
                connections.connect(alias=self.connection_alias, host=self.host, port=self.port)
        except Exception as e:
            _log.error(f"Failed to connect to Milvus: {e}")
            raise

    def _resolve_embedder(self, model):
        if model is None:
            return None
        if isinstance(model, str) and OpenAIEmbeddings:
             # Basic default factory using settings for keys if not provided
             return OpenAIEmbeddings(
                model=model,
                openai_api_base=settings.llm.api_base,
                openai_api_key=settings.llm.api_key
             )
        return model

    def _resolve_reranker(self, model):
        if model is None:
            return None
        if isinstance(model, str) and RerankerWrapper:
             # Basic default factory
             return RerankerWrapper(model_name=model)
        return model

    def _init_collection(self, overwrite: bool) -> Collection:
        if utility.has_collection(self.collection_name, using=self.connection_alias):
            if overwrite:
                utility.drop_collection(self.collection_name, using=self.connection_alias)
                _log.info(f"Dropped existing collection: {self.collection_name}")
                return self._create_collection()
            else:
                _log.info(f"Loaded existing collection: {self.collection_name}")
                return Collection(self.collection_name, using=self.connection_alias)
        else:
            return self._create_collection()

    def _create_collection(self) -> Collection:
        _log.info(f"Creating new collection: {self.collection_name}")
        return Collection(
            name=self.collection_name,
            schema=self.schema,
            using=self.connection_alias,
            consistency_level="Strong"
        )

    def _create_index(self):
        index_params = {
            "index_type": "IVF_FLAT",
            "metric_type": "COSINE", 
            "params": {"nlist": 128}
        }
        _log.info(f"Creating index for {self.collection_name}...")
        self.collection.create_index(
            field_name="embedding",
            index_params=index_params
        )
        # Sparse Index
        sparse_index_params = {
            "index_type": "SPARSE_INVERTED_INDEX",
            "metric_type": "IP",
            "params": {"drop_ratio_build": 0.2}
        }
        self.collection.create_index(
            field_name="sparse_vector",
            index_params=sparse_index_params
        )
        self.collection.load()

    def insert(self, documents: List[Document], batch_size: int = 64):
        """
        Insert documents. 
        If document has 'embedding' in metadata, use it.
        Otherwise, generate embedding using self.embedder.
        """
        total = len(documents)
        if total == 0:
            return

        _log.info(f"Inserting {total} documents into {self.collection_name}...")
        
        for i in range(0, total, batch_size):
            batch = documents[i:i + batch_size]
            
            # Prepare data columns
            texts_col = []
            embeds_col = []
            sparse_col = []
            metas_col = []
            lengths_col = []
            
            # Need to compute embeddings for docs that miss them?
            docs_needing_embed = []
            indices_needing_embed = []
            
            for idx, doc in enumerate(batch):
                emb = doc.metadata.get("embedding")
                
                if emb is None:
                    if self.embedder:
                        docs_needing_embed.append(doc.page_content)
                        indices_needing_embed.append(idx)
                        embeds_col.append(None) # Place holder
                    else:
                        raise ValueError(f"Document {idx} missing embedding and no embedder configured.")
                else:
                    # Normalized to list
                    if hasattr(emb, "tolist"):
                        emb = emb.tolist()
                    embeds_col.append(emb)
                
                # Sparse Handling (Optional in docs)
                sparse = doc.metadata.get("sparse_vector", {})
                sparse_col.append(sparse)
                
                texts_col.append(doc.page_content)
                lengths_col.append(len(doc.page_content))
                
                # Clean metadata (remove embedding to avoid dup)
                clean_meta = doc.metadata.copy()
                if "embedding" in clean_meta:
                    del clean_meta["embedding"]
                metas_col.append(clean_meta)

            # Batch compute embeddings if needed
            if docs_needing_embed:
                _log.info(f"Computing embeddings for {len(docs_needing_embed)} documents...")
                computed_embeds = self.embedder.embed_documents(docs_needing_embed)
                for map_idx, real_idx in enumerate(indices_needing_embed):
                    embeds_col[real_idx] = computed_embeds[map_idx]

            # Validate dimensions
            for emb in embeds_col:
                if len(emb) != self.dim:
                     raise ValueError(f"Embedding dimension mismatch: got {len(emb)}, expected {self.dim}")

            # Insert
            entities = [
                texts_col,
                embeds_col,
                sparse_col,
                metas_col,
                lengths_col
            ]
            self.collection.insert(entities)
            _log.info(f"Inserted batch {i}-{min(i+batch_size, total)}")

        self.collection.flush()
        _log.info("Insertion complete.")

    def search(
        self,
        query: str,
        top_k: int = 5,
        rerank: bool = False,
        score_threshold: float = 0.0,
        filter_expr: str = ""
    ) -> List[Document]:
        
        if not self.embedder:
            raise ValueError("No embedder configured for search.")

        query_vec = self.embedder.embed_query(query)
        
        search_params = {
            "metric_type": "COSINE",
            "params": {"nprobe": 10},
        }
        
        # Search
        res = self.collection.search(
            data=[query_vec],
            anns_field="embedding",
            param=search_params,
            limit=top_k * 2 if rerank else top_k, # Fetch more for reranking
            expr=filter_expr,
            output_fields=["text", "metadata"],
            consistency_level="Strong"
        )
        
        results = []
        for hits in res:
            for hit in hits:
                # Milvus distance (Cosine) might be distance or similarity depending on config
                # Usually COSINE metric in Milvus returns distance? No, IP is inner product. 
                # If metric is COSINE, pymilvus returns distance. 
                # But widely, we treat 1 - distance or score.
                # Let's assume score is similarity for now if metric is COSINE (range 0-1 if normalized) via "score" attribute.
                score = hit.distance # In pymilvus search result, .distance is the score.
                
                if score < score_threshold:
                    continue

                doc = Document(
                    page_content=hit.entity.get("text"),
                    metadata=hit.entity.get("metadata", {})
                )
                doc.metadata["score"] = score
                results.append(doc)
        
                doc.metadata["score"] = score
                results.append(doc)
        
        # Rerank
        if rerank and self.reranker:
            return self._rerank(query, results, top_k)
            
        return results[:top_k]

    def hybrid_search(
        self,
        query: str,
        sparse_embedding: Dict[int, float], # Generated by BGE-M3
        top_k: int = 5,
        rerank: bool = False,
        dense_weight: float = 1.0,
        sparse_weight: float = 0.3
    ) -> List[Document]:
        """
        Perform Hybrid Search (Dense + Sparse) with Weighted Reranking (RRFRanker or WeightedRanker).
        Milvus support Hybrid Search via `hybrid_search` but pure python client usually requires
        doing 2 searches and fusing. 
        Actually Milvus 2.4 has `hybrid_search` API.
        Alternatively, we can use `AnnSearchRequest` with `RRFRanker`.
        """
        if not self.embedder:
             raise ValueError("No embedder.")

        query_vec = self.embedder.embed_query(query)
        
        # Dense Request
        from pymilvus import AnnSearchRequest, RRFRanker, WeightedRanker
        
        dense_req = AnnSearchRequest(
            data=[query_vec],
            anns_field="embedding",
            param={"metric_type": "COSINE", "params": {"nprobe": 10}},
            limit=top_k * 2
        )
        
        # Sparse Request
        if not sparse_embedding:
             # Fallback to dense if no sparse
             return self.search(query, top_k, rerank)

        sparse_req = AnnSearchRequest(
            data=[sparse_embedding],
            anns_field="sparse_vector",
            param={"metric_type": "IP", "params": {"drop_ratio_build": 0.2}}, # Sparse params
            limit=top_k * 2
        )
        
        # Reranker Strategy
        # ranker = RRFRanker(k=60)
        ranker = WeightedRanker(dense_weight, sparse_weight)
        
        try:
            res = self.collection.hybrid_search(
                reqs=[dense_req, sparse_req],
                ranker=ranker,
                limit=top_k * 2 if rerank else top_k,
                output_fields=["text", "metadata"],
                consistency_level="Strong"
            )
        except Exception as e:
            _log.error(f"Hybrid search failed: {e}. Fallback to dense.")
            return self.search(query, top_k, rerank)
            
        results = []
        for hits in res:
             for hit in hits:
                doc = Document(
                    page_content=hit.entity.get("text"),
                    metadata=hit.entity.get("metadata", {})
                )
                doc.metadata["score"] = hit.score
                results.append(doc)

        if rerank and self.reranker:
            return self._rerank(query, results, top_k)
        
        return results[:top_k]

    def get_adjacent_chunks(self, file_id: str, center_index: int, radius: int = 1) -> List[Document]:
        """
        Retrieve chunks adjacent to the center index for a specific file.
        Used for Context Window Expansion (Parent Document Retrieval approximation).
        """
        min_idx = max(0, center_index - radius)
        max_idx = center_index + radius
        
        # Milvus JSON Filter
        # Note: Milvus Lite might have limited JSON support, but assuming standard syntax.
        # expr = f'metadata["file_id"] == "{file_id}" && metadata["chunk_index"] >= {min_idx} && metadata["chunk_index"] <= {max_idx}'
        # Pymilvus expression syntax for JSON field:
        expr = f'metadata["file_id"] == "{file_id}" and metadata["chunk_index"] >= {min_idx} and metadata["chunk_index"] <= {max_idx}'
        
        try:
            res = self.collection.query(
                expr=expr,
                output_fields=["text", "metadata"],
                consistency_level="Strong"
            )
        except Exception as e:
            _log.error(f"Failed to query adjacent chunks: {e}")
            return []
            
        docs = []
        for hit in res:
             doc = Document(
                 page_content=hit.get("text"),
                 metadata=hit.get("metadata", {})
             )
             docs.append(doc)
        
        # Sort by chunk_index
        docs.sort(key=lambda x: x.metadata.get("chunk_index", 0))
        return docs
        if not docs:
            return []
            
        doc_texts = [d.page_content for d in docs]
        scores = self.reranker.compute_score(query, doc_texts, normalize=True)
        # RerankerWrapper might return list of floats
        
        for d, s in zip(docs, scores):
            d.metadata["rerank_score"] = float(s)
            
        # Sort by rerank score
        docs.sort(key=lambda x: x.metadata["rerank_score"], reverse=True)
        return docs[:top_k]

    def close(self):
        self.collection.release()
        connections.disconnect(self.connection_alias)
