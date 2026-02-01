import os
import shutil
import time
import traceback
from typing import Any

from pymilvus import CollectionSchema, DataType, FieldSchema, MilvusClient, MilvusException

from src.core.feature_flags import feature_enabled
from src.core.settings import settings
from src.knowledge.store.kb_db import kb_db_manager
from src.utils import hashstr
from src.utils.logger import get_logger

logger = get_logger(__name__)


# 知识库管理
class KnowledgeBase:
    """
    集成文档分块、向量化、Milvus 存储、检索、知识库管理的全能版

    功能：
    - 知识库管理：创建/删除 库，获取库列表和详情
    - 文档导入：单文件/目录导入，支持多格式 & OCR
    - 向量存储：Milvus Collection 管理，插入文档向量
    - 检索：基于向量相似度和可选重排序
    - 文件管理：记录文件状态，支持批量迁移、备份
    """

    def __init__(self, milvus_uri: str | None = None, embedding_config: dict[str, Any] | None = None) -> None:
        self._milvus_uri = milvus_uri
        self._embedding_config = embedding_config

        # 工作目录 & DB 管理
        self.work_dir = os.path.join(str(settings.paths.save_yaml_path), "data")
        os.makedirs(self.work_dir, exist_ok=True)
        self.db_manager = kb_db_manager

        # 检索参数
        self.default_distance_threshold = settings.kb_config.default_distance_threshold
        self.default_rerank_threshold = settings.kb_config.default_rerank_threshold
        self.default_max_query_count = settings.kb_config.default_max_query_count
        self.top_k = settings.kb_config.default_top_k
        self.conf = ""

        # 运行时依赖（懒加载）
        self.client: MilvusClient | None = None
        self.embed_model = None
        self.reranker = None

        # 初始化（轻量）
        self._check_migration()
        # 注意：Milvus/Embedding 初始化会阻塞/失败（服务未启动、未配置 key 等），
        # 因此这里改为懒加载，直到真正调用 KB 相关能力时再连接。

    def _is_enabled(self) -> bool:
        # Allow callers to force-enable via embedding_config, otherwise respect runtime feature flags.
        if isinstance(self._embedding_config, dict) and "enable_knowledge_base" in self._embedding_config:
            return bool(self._embedding_config.get("enable_knowledge_base"))
        return bool(feature_enabled("enable_knowledge_base"))

    # -- 数据迁移 -----------------------------------------------------------
    def _check_migration(self):
        legacy = os.path.join(self.work_dir, "database.json")
        if os.path.exists(legacy):
            logger.info("检测到旧 JSON 知识库，迁移中...")
            try:
                from scripts.migrate_kb_to_sqlite import migrate_json_to_sqlite

                migrate_json_to_sqlite()
                logger.info("迁移完成！")
            except Exception as e:
                logger.error(f"迁移失败: {e}")

    # -- Embedding 模型 ---------------------------------------------------
    def _load_embedding_model(self, embedding_config: dict[str, Any] | None):
        logger.info(f"传入的 embedding_config: {embedding_config}")
        if not self._is_enabled():
            self.embed_model = None
            self.reranker = None
            return

        from src.models.embedding import get_embedding_model

        # 使用新的 settings 获取 embedding 配置
        # model_name格式: "BAAI/bge-m3" 或带provider前缀
        model_name = settings.embedding.model_name

        # 如果传入了自定义配置，优先使用
        if embedding_config and isinstance(embedding_config, dict):
            embed_model_str = embedding_config.get("embed_model", "")
            if embed_model_str:
                model_name = embed_model_str

        self.conf = model_name
        self.embed_model = get_embedding_model(model=model_name)

        if feature_enabled("enable_reranker"):
            from src.models.reranker_model import get_reranker

            self.reranker = get_reranker()
        else:
            self.reranker = None

    # -- Milvus 连接 --------------------------------------------------------
    def _connect_milvus(self, uri: str | None):
        try:
            # 优先使用函数参数 -> 再看环境变量 -> 再看 settings 默认值
            target = uri or os.getenv("MILVUS_URI") or settings.database.milvus_uri

            # 自动补 http:// 前缀，防止只写了 localhost:19530
            if not target.startswith("http://") and not target.startswith("https://"):
                target = "http://" + target

            self.client = MilvusClient(uri=target)
            self.client.list_collections()
            logger.info(f"Milvus 已连接: {target}")
        except MilvusException as e:
            logger.error(f"连接 Milvus 失败: {e}")
            raise

    def _ensure_ready(self):
        """
        Ensure KB runtime dependencies are initialized.
        This keeps module import cheap and prevents server startup from blocking.
        """
        if not self._is_enabled():
            raise RuntimeError("KnowledgeBase is disabled (enable_knowledge_base=false).")

        if self.embed_model is None:
            self._load_embedding_model(self._embedding_config)

        if self.client is None:
            self._connect_milvus(self._milvus_uri)

    # -- 知识库管理 --------------------------------------------------------
    def create_database(self, name: str, description: str, dimension: int | None = None) -> dict[str, Any]:
        self._ensure_ready()
        dim = dimension or self.embed_model.get_dimension()
        db_id = f"kb_{hashstr(name)}"
        info = self.db_manager.create_database(
            db_id=db_id, name=name, description=description, embed_model=self.conf, dimension=dim
        )
        self._ensure_directories(db_id)
        self.add_collection(db_id, dim)
        return info

    def delete_database(self, db_id: str) -> None:
        if self._is_enabled():
            # Best-effort Milvus cleanup; still delete local metadata even if Milvus isn't reachable.
            try:
                self._ensure_ready()
                if self.client.has_collection(db_id):
                    self.client.drop_collection(db_id)
            except Exception as e:
                logger.warning(f"Drop Milvus collection failed (ignored): {e}")
        self.db_manager.delete_database(db_id)
        folder = os.path.join(self.work_dir, db_id)
        if os.path.isdir(folder):
            shutil.rmtree(folder)

    def list_databases(self) -> list[dict[str, Any]]:
        out = []
        for db in self.db_manager.get_all_databases():
            record = db.copy()
            try:
                record["metadata"] = self.get_collection_info(db["db_id"])
            except Exception:
                record["metadata"] = {"error": "无法获取"}
            out.append(record)
        return out

    # -- 文件与文档导入 ----------------------------------------------------
    def ingest_file(
        self,
        db_id: str,
        path: str,
        do_ocr: bool = False,
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
        ocr_det_threshold: float = 0.3,
    ) -> str:
        """
        导入单个文件：
        - PDF 可选 OCR
        - 其他文本使用 chunk & read_text
        - 自动记录并插入 Milvus
        返回 file_id
        """
        ext = path.split(".")[-1].lower()
        file_id = f"file_{hashstr(path + str(time.time()))}"
        _, upload_folder = self._ensure_directories(db_id)
        os.makedirs(upload_folder, exist_ok=True)
        if not os.path.isabs(path):
            base_dir, _ = self._ensure_directories(db_id)
            norm_path = os.path.normpath(path)
            norm_path = norm_path.replace("\\", "/")
            if ".." in norm_path or norm_path.startswith("/"):
                raise ValueError(f"非法路径: {path}")
            path = os.path.join(base_dir, norm_path)

        # 分块
        from src.knowledge.core.indexing import chunk_file

        try:
            if ext == "pdf" or do_ocr:
                docs = chunk_file(path, chunk_size, chunk_overlap, True, ocr_det_threshold)
            else:
                docs = chunk_file(path, chunk_size, chunk_overlap)
        except Exception as e:
            logger.error(f"分块失败: {e}")
            raise

        texts = [d.page_content for d in docs]

        # 数据库记录
        self.db_manager.add_file(
            db_id=db_id, file_id=file_id, filename=os.path.basename(path), path=path, file_type=ext, status="processing"
        )

        # 向量插入
        try:
            self._ensure_ready()
            chunks = [d.metadata | {"text": d.page_content} for d in docs]
            self._insert_vectors(db_id, file_id, texts, chunks)
            self.db_manager.update_file_status(file_id, "done")
        except Exception as e:
            logger.error(f"向量插入失败: {e}\n{traceback.format_exc()}")
            self.db_manager.update_file_status(file_id, "failed")

        return file_id

    def ingest_directory(self, db_id: str, folder: str, suffixes: list[str] | None = None, **kwargs) -> list[str]:
        suffixes = suffixes or [".pdf", ".txt", ".md", ".docx"]
        ids = []
        for root, _, files in os.walk(folder):
            for f in files:
                if any(f.lower().endswith(s) for s in suffixes):
                    path = os.path.join(root, f)
                    fid = self.ingest_file(db_id, path, **kwargs)
                    ids.append(fid)
        return ids

    def _ensure_directories(self, db_id: str) -> (str, str):
        base = os.path.join(self.work_dir, db_id)
        upload = os.path.join(base, "uploads")
        os.makedirs(base, exist_ok=True)
        os.makedirs(upload, exist_ok=True)
        return base, upload

    # -- Milvus Collection 操作 --------------------
    def add_collection(self, name: str, dimension: int) -> None:
        self._ensure_ready()
        if self.client.has_collection(name):
            self.client.drop_collection(name)

        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=128),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dimension),
            FieldSchema(name="file_id", dtype=DataType.VARCHAR, max_length=128),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
        ]
        schema = CollectionSchema(fields=fields, description="KB vector schema")
        self.client.create_collection(collection_name=name, schema=schema)

        from pymilvus import Collection, connections

        # MilvusClient does not auto-register the default ORM connection used by Collection.
        # Ensure the default alias exists before creating index/load.
        if not connections.has_connection("default"):
            target = self._milvus_uri or os.getenv("MILVUS_URI") or settings.database.milvus_uri
            if not target.startswith("http://") and not target.startswith("https://"):
                target = "http://" + target
            connections.connect(alias="default", uri=target)

        collection = Collection(name, using="default")
        collection.create_index(
            field_name="vector", index_params={"index_type": "IVF_FLAT", "metric_type": "L2", "params": {"nlist": 1024}}
        )
        collection.load()

    def get_collection_info(self, name: str) -> dict[str, Any]:
        try:
            self._ensure_ready()
            info = self.client.describe_collection(name)
            info.update(self.client.get_collection_stats(name))
            return info
        except MilvusException as e:
            return {"name": name, "error": str(e)}

    def _insert_vectors(
        self, collection_name: str, file_id: str, docs: list[str], chunk_infos: list[dict[str, Any]]
    ) -> Any:
        self._ensure_ready()
        if not self.client.has_collection(collection_name):
            raise ValueError("Collection不存在")

        vecs = self.embed_model.batch_encode(docs)

        entities = []
        for idx, v in enumerate(vecs):
            meta = chunk_infos[idx]
            meta["file_id"] = file_id
            meta["text"] = docs[idx]
            vector_id = f"{file_id}_{idx}"
            entities.append({"id": vector_id, "vector": v, "file_id": meta["file_id"], "text": meta["text"]})

        return self.client.insert(collection_name=collection_name, data=entities)

    # -- 检索 --------------------------------------------------------------
    def search(
        self,
        query: str,
        db_id: str,
        distance_threshold: float | None = None,
        rerank: bool = True,
        top_k: int | None = None,
    ) -> dict[str, Any]:
        self._ensure_ready()
        dt = distance_threshold if distance_threshold is not None else self.default_distance_threshold
        tk = top_k if top_k is not None else self.top_k

        #  向量化查询
        vector = self.embed_model.batch_encode([query])[0]

        #  Milvus 检索
        raw_res = self.client.search(
            db_id, [vector], limit=self.default_max_query_count, output_fields=["text", "file_id"]
        )

        # 转成纯Python可序列化结构
        hits: list[dict[str, Any]] = raw_res[0]
        results: list[dict[str, Any]] = []
        for h in hits:
            # MilvusClient.search() 返回 {'id': ..., 'distance': ..., 'entity': {'text': ..., 'file_id': ...}}
            entity = h.get("entity", {})
            results.append(
                {
                    "entity": {
                        "text": entity.get("text", ""),
                        "file_id": entity.get("file_id"),
                        "id": h.get("id"),
                    },
                    "distance": h.get("distance", h.get("score", 0.0)),
                }
            )

        # 距离阈值过滤 - 当 dt >= 1.0 时禁用过滤（用于调试）
        if dt >= 1.0:
            filtered = results
        else:
            filtered = [r for r in results if r["distance"] < dt]

        # 重排序（带错误处理）
        if rerank and self.reranker and filtered:
            try:
                texts = [r["entity"]["text"] for r in filtered]
                scores = self.reranker.compute_score(query, texts, normalize=False)
                for r, s in zip(filtered, scores):
                    r["rerank_score"] = float(s)  # 转 float，保证可 JSON
                # 只排序，不过滤 - 让用户看到所有结果
                filtered.sort(key=lambda x: x["rerank_score"], reverse=True)
            except Exception as e:
                logger.warning(f"Rerank failed, using distance order: {e}")
                # 失败时按距离排序
                filtered.sort(key=lambda x: x["distance"])

        return {"results": filtered[:tk], "all_results": results}

    def restart(self):
        # Reset runtime clients (lazy rebuild).
        self.client = None
        self.embed_model = None
        self.reranker = None
        # Keep previous config but delay actual connection until first use.
