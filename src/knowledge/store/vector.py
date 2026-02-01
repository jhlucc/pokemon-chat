import warnings
from typing import Any

from langchain_core.documents import Document
from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)

from src.core.settings import settings
from src.utils.logger import get_logger
from src.utils.net import parse_host_port

try:
    from src.models.reranker_model import RerankerWrapper
except ImportError:
    RerankerWrapper = None

warnings.filterwarnings("ignore", category=FutureWarning)
_log = get_logger(__name__)


class _EmbeddingAdapter:
    """
    Adapter for the project's embedding models.

    The VectorStore expects an object that implements:
      - embed_documents(list[str]) -> list[list[float]]
      - embed_query(str) -> list[float]
    """

    def __init__(self, model: Any):
        self._model = model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._model.embed(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._model.embed(text)[0]


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
        embedding_model: Any | None = None,  # Can be instance or name
        reranker_model: Any | None = None,  # Can be instance or name
        connection_alias: str = "default",
        enable_sparse: bool = False,  # Disable sparse vectors by default (compatibility)
    ):
        self.collection_name = collection_name
        self.dim = dim
        self.host = host
        self.port = port
        self.connection_alias = connection_alias
        self.enable_sparse = enable_sparse

        # If the caller kept the default host/port, prefer the project's unified setting
        # (supports Docker service names like "milvus").
        if (host, str(port)) == ("localhost", "19530"):
            try:
                uri = (getattr(settings.database, "milvus_uri", "") or "").strip()
                if uri:
                    resolved_host, resolved_port = parse_host_port(uri, default_port=19530)
                    self.host = resolved_host
                    self.port = str(resolved_port)
            except Exception:  # noqa: BLE001
                pass

        # Initialize models (optional, for on-the-fly embedding/reranking)
        self.embedder = self._resolve_embedder(embedding_model)
        self.reranker = self._resolve_reranker(reranker_model)

        # Connect to Milvus
        self._connect()

        # Schemas - sparse vector field is optional
        self.fields = [
            FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
            FieldSchema(name="metadata", dtype=DataType.JSON),
            FieldSchema(name="text_length", dtype=DataType.INT64),
        ]
        if self.enable_sparse:
            self.fields.append(FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR))

        self.schema = CollectionSchema(
            fields=self.fields, description="Knowledge Base Documents", enable_dynamic_field=True
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
        if isinstance(model, str):
            # Prefer the project's embedding factory so provider-specific endpoints
            # (e.g. SiliconFlow) work with the same `.env` settings everywhere.
            try:
                from src.models.embedding import get_embedding_model

                return _EmbeddingAdapter(get_embedding_model(model=model))
            except Exception as e:  # noqa: BLE001
                _log.warning(f"Embedding model init failed (will disable embedding): {e}")
                return None

        # Accept the project's embedding model instances directly.
        if hasattr(model, "embed") and not hasattr(model, "embed_documents"):
            return _EmbeddingAdapter(model)
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
            name=self.collection_name, schema=self.schema, using=self.connection_alias, consistency_level="Strong"
        )

    def _create_index(self):
        index_params = {"index_type": "IVF_FLAT", "metric_type": "COSINE", "params": {"nlist": 128}}
        _log.info(f"Creating index for {self.collection_name}...")
        self.collection.create_index(field_name="embedding", index_params=index_params)

        # Sparse Index (only if enabled)
        if self.enable_sparse:
            try:
                sparse_index_params = {
                    "index_type": "SPARSE_INVERTED_INDEX",
                    "metric_type": "IP",
                    "params": {"drop_ratio_build": 0.2},
                }
                self.collection.create_index(field_name="sparse_vector", index_params=sparse_index_params)
                _log.info("Sparse vector index created successfully")
            except Exception as e:
                _log.warning(f"Sparse vector index creation failed: {e}")

        self.collection.load()

    def insert(self, documents: list[Document], batch_size: int = 64):
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
            batch = documents[i : i + batch_size]

            # Prepare data columns
            texts_col = []
            embeds_col = []
            sparse_col = [] if self.enable_sparse else None
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
                        embeds_col.append(None)  # Place holder
                    else:
                        raise ValueError(f"Document {idx} missing embedding and no embedder configured.")
                else:
                    # Normalized to list
                    if hasattr(emb, "tolist"):
                        emb = emb.tolist()
                    embeds_col.append(emb)

                # Sparse Handling (only if enabled)
                if self.enable_sparse:
                    sparse = doc.metadata.get("sparse_vector", {})
                    sparse_col.append(sparse)

                texts_col.append(doc.page_content)
                lengths_col.append(len(doc.page_content))

                # Clean metadata (remove embedding to avoid dup)
                clean_meta = doc.metadata.copy()
                if "embedding" in clean_meta:
                    del clean_meta["embedding"]
                if "sparse_vector" in clean_meta:
                    del clean_meta["sparse_vector"]
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

            # Insert - build entities list based on schema
            if self.enable_sparse:
                entities = [texts_col, embeds_col, sparse_col, metas_col, lengths_col]
            else:
                entities = [texts_col, embeds_col, metas_col, lengths_col]
            self.collection.insert(entities)
            _log.info(f"Inserted batch {i}-{min(i + batch_size, total)}")

        self.collection.flush()
        _log.info("Insertion complete.")

    def search(
        self, query: str, top_k: int = 5, rerank: bool = False, score_threshold: float = 0.0, filter_expr: str = ""
    ) -> list[Document]:
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
            limit=top_k * 2 if rerank else top_k,  # Fetch more for reranking
            expr=filter_expr,
            output_fields=["text", "metadata"],
            consistency_level="Strong",
        )

        results = []
        for hits in res:
            for hit in hits:
                # Milvus distance (Cosine) might be distance or similarity depending on config
                # Usually COSINE metric in Milvus returns distance? No, IP is inner product.
                # If metric is COSINE, pymilvus returns distance.
                # But widely, we treat 1 - distance or score.
                # Let's assume score is similarity for now if metric is COSINE (range 0-1 if normalized) via "score" attribute.
                score = hit.distance  # In pymilvus search result, .distance is the score.

                if score < score_threshold:
                    continue

                # Access entity fields - compatible with different pymilvus versions
                entity = hit.entity
                text = entity.get("text") if hasattr(entity, "get") else entity["text"]
                metadata = entity.get("metadata") if hasattr(entity, "get") else entity.get("metadata", {})
                if metadata is None:
                    metadata = {}

                doc = Document(page_content=text, metadata=metadata)
                doc.metadata["score"] = score
                results.append(doc)

        # Rerank
        if rerank and self.reranker:
            return self._rerank(query, results, top_k)

        return results[:top_k]

    def _rerank(self, query: str, docs: list[Document], top_k: int) -> list[Document]:
        """
        Rerank retrieved docs using the configured reranker.

        Notes:
        - This is best-effort: if reranking fails, we fall back to the original order.
        - We keep the original vector score under `score` and store the rerank score
          under `rerank_score`.
        """
        if not docs:
            return []
        if not self.reranker:
            return docs[:top_k]

        try:
            scores = self.reranker.run(query, [d.page_content for d in docs], normalize=True)
        except Exception as e:  # noqa: BLE001
            _log.warning(f"Rerank failed; returning original order: {e}")
            return docs[:top_k]

        # Attach rerank scores (best-effort).
        for doc, score in zip(docs, scores, strict=False):
            try:
                doc.metadata["rerank_score"] = float(score)
            except Exception:
                doc.metadata["rerank_score"] = score

        # Optional filtering by threshold (avoid returning low-confidence docs).
        threshold = float(getattr(settings.kb_config, "default_rerank_threshold", 0.0) or 0.0)
        filtered = (
            [d for d in docs if float(d.metadata.get("rerank_score") or 0.0) >= threshold]
            if threshold > 0
            else list(docs)
        )
        if not filtered:
            filtered = list(docs)

        filtered.sort(key=lambda d: float(d.metadata.get("rerank_score") or 0.0), reverse=True)
        return filtered[:top_k]

    def hybrid_search(
        self,
        query: str,
        sparse_embedding: dict[int, float],  # Generated by BGE-M3
        top_k: int = 5,
        rerank: bool = False,
        dense_weight: float = 1.0,
        sparse_weight: float = 0.3,
    ) -> list[Document]:
        """
        Perform Hybrid Search (Dense + Sparse) with Weighted Reranking (RRFRanker or WeightedRanker).
        Requires enable_sparse=True in constructor for real Milvus sparse-field support.

        Note: if a sparse embedding is provided but the store was initialized without
        sparse support, we still *attempt* hybrid_search (it may fail and fall back).
        """
        if not self.enable_sparse:
            if not sparse_embedding:
                _log.warning(
                    "Hybrid search called but sparse vectors not enabled and no sparse embedding provided. "
                    "Falling back to dense search."
                )
                return self.search(query, top_k, rerank)
            _log.warning(
                "Hybrid search called but sparse vectors not enabled. "
                "Attempting hybrid_search anyway because a sparse embedding was provided (may fail)."
            )

        if not self.embedder:
            raise ValueError("No embedder.")

        query_vec = self.embedder.embed_query(query)

        # Dense Request
        from pymilvus import AnnSearchRequest, WeightedRanker

        dense_req = AnnSearchRequest(
            data=[query_vec],
            anns_field="embedding",
            param={"metric_type": "COSINE", "params": {"nprobe": 10}},
            limit=top_k * 2,
        )

        # Sparse Request
        if not sparse_embedding:
            # Fallback to dense if no sparse
            return self.search(query, top_k, rerank)

        sparse_req = AnnSearchRequest(
            data=[sparse_embedding],
            anns_field="sparse_vector",
            param={"metric_type": "IP", "params": {"drop_ratio_build": 0.2}},  # Sparse params
            limit=top_k * 2,
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
                consistency_level="Strong",
            )
        except Exception as e:
            _log.error(f"Hybrid search failed: {e}. Fallback to dense.")
            return self.search(query, top_k, rerank)

        results = []
        for hits in res:
            for hit in hits:
                entity = hit.entity
                text = entity.get("text") if hasattr(entity, "get") else entity["text"]
                metadata = entity.get("metadata") if hasattr(entity, "get") else entity.get("metadata", {})
                if metadata is None:
                    metadata = {}
                doc = Document(page_content=text, metadata=metadata)
                doc.metadata["score"] = hit.score
                results.append(doc)

        if rerank and self.reranker:
            return self._rerank(query, results, top_k)

        return results[:top_k]

    def get_adjacent_chunks(self, file_id: str, center_index: int, radius: int = 1) -> list[Document]:
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
            res = self.collection.query(expr=expr, output_fields=["text", "metadata"], consistency_level="Strong")
        except Exception as e:
            _log.error(f"Failed to query adjacent chunks: {e}")
            return []

        docs = []
        for hit in res:
            doc = Document(
                page_content=hit.get("text"),
                metadata=hit.get("metadata", {}),
            )
            docs.append(doc)

        # Sort by chunk_index
        docs.sort(key=lambda x: x.metadata.get("chunk_index", 0))
        return docs

    def close(self):
        self.collection.release()
        connections.disconnect(self.connection_alias)
