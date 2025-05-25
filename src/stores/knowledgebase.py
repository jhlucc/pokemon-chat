import os
import shutil
import time
import traceback
import random
from typing import List, Optional, Dict, Any

from pymilvus import MilvusClient, MilvusException
from pymilvus import FieldSchema, CollectionSchema, DataType
from src import config
from src.utils import  hashstr

from src.stores.kb_db_manager import kb_db_manager
from src.utils.logger import LogManager
logger= LogManager()
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

    def __init__(
        self,
        milvus_uri: Optional[str] = None,
        embedding_config: Optional[Dict[str, Any]] = None
    ) -> None:
        # 工作目录 & DB 管理
        self.work_dir = os.path.join(config.save_dir, "data")
        os.makedirs(self.work_dir, exist_ok=True)
        self.db_manager = kb_db_manager

        # 检索参数
        self.default_distance_threshold = config.get("default_distance_threshold", 0.5)
        self.default_rerank_threshold = config.get("default_rerank_threshold", 0.1)
        self.default_max_query_count = config.get("default_max_query_count", 20)
        self.top_k = config.get("default_top_k", 10)
        self.conf=0
        # 初始化模型与服务
        self._check_migration()
        self._load_embedding_model(embedding_config)
        self._connect_milvus(milvus_uri)

    # -- 数据迁移 -----------------------------------------------------------
    def _check_migration(self):
        legacy = os.path.join(self.work_dir, "database.json")
        if os.path.exists(legacy):
            logger.info("检测到旧 JSON 知识库，迁移中...")
            try:
                from src.stores.migrate_kb_to_sqlite import migrate_json_to_sqlite
                migrate_json_to_sqlite()
                logger.info("迁移完成！")
            except Exception as e:
                logger.error(f"迁移失败: {e}")

    # -- Embedding 模型 ---------------------------------------------------
    def _load_embedding_model(self, embedding_config: Optional[Dict[str, Any]]):
        logger.info(f"传入的 embedding_config: {embedding_config}")
        if not config.enable_knowledge_base:
            self.embed_model = None
            return
        from src.models.embedding import get_embedding_model

        # conf ="local/bge-large-zh-v1.5" or embedding_config or config
        conf = embedding_config or {
            "enable_knowledge_base": True,
            "embed_model": "local/bge-large-zh-v1.5"
        }
        # self.conf = "local/bge-large-zh-v1.5"
        self.conf = conf["embed_model"]  # 记录模型名称字符串
        self.embed_model = get_embedding_model(conf)
        if config.enable_reranker:
            from src.models.reranker_model import  RerankerWrapper
            self.reranker = RerankerWrapper("siliconflow/bge-reranker-v2-m3", model_name="BAAI/bge-reranker-v2-m3")
        else:
            self.reranker = None

    # -- Milvus 连接 --------------------------------------------------------
    def _connect_milvus(self, uri: Optional[str]):
        try:
            # 优先使用函数参数 -> 再看环境变量 -> 再看 config 默认值
            target = uri or os.getenv("MILVUS_URI") or config.get("milvus_uri", "http://localhost:19530")

            # 自动补 http:// 前缀，防止只写了 localhost:19530
            if not target.startswith("http://") and not target.startswith("https://"):
                target = "http://" + target

            self.client = MilvusClient(uri=target)
            self.client.list_collections()
            logger.info(f"Milvus 已连接: {target}")
        except MilvusException as e:
            logger.error(f"连接 Milvus 失败: {e}")
            raise

    # -- 知识库管理 --------------------------------------------------------
    def create_database(self, name: str, description: str, dimension: Optional[int] = None) -> Dict[str, Any]:
        dim = dimension or self.embed_model.get_dimension()
        db_id = f"kb_{hashstr(name)}"
        info = self.db_manager.create_database(
            db_id=db_id,
            name=name,
            description=description,
            embed_model=self.conf,
            dimension=dim
        )
        self._ensure_directories(db_id)
        self.add_collection(db_id, dim)
        return info

    def delete_database(self, db_id: str) -> None:
        if self.client.has_collection(db_id):
            self.client.drop_collection(db_id)
        self.db_manager.delete_database(db_id)
        folder = os.path.join(self.work_dir, db_id)
        if os.path.isdir(folder): shutil.rmtree(folder)

    def list_databases(self) -> List[Dict[str, Any]]:
        out = []
        for db in self.db_manager.get_all_databases():
            record = db.copy()
            try:
                record['metadata'] = self.get_collection_info(db['db_id'])
            except Exception:
                record['metadata'] = {'error': '无法获取'}
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
        ocr_det_threshold: float = 0.3
    ) -> str:
        """
        导入单个文件：
        - PDF 可选 OCR
        - 其他文本使用 chunk & read_text
        - 自动记录并插入 Milvus
        返回 file_id
        """
        ext = path.split('.')[-1].lower()
        file_id = f"file_{hashstr(path+str(time.time()))}"
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
        from rag.core.indexing import chunk_file
        try:
            if ext == 'pdf' or do_ocr:
                docs = chunk_file(path, chunk_size, chunk_overlap, True, ocr_det_threshold)
            else:
                docs = chunk_file(path, chunk_size, chunk_overlap)
        except Exception as e:
            logger.error(f"分块失败: {e}")
            raise

        texts = [d.page_content for d in docs]

        # 数据库记录
        self.db_manager.add_file(
            db_id=db_id,
            file_id=file_id,
            filename=os.path.basename(path),
            path=path,
            file_type=ext,
            status='processing'
        )

        # 向量插入
        try:
            chunks = [d.metadata | {"text": d.page_content} for d in docs]
            self._insert_vectors(db_id, file_id, texts, chunks)
            self.db_manager.update_file_status(file_id, 'done')
        except Exception as e:
            logger.error(f"向量插入失败: {e}\n{traceback.format_exc()}")
            self.db_manager.update_file_status(file_id, 'failed')

        return file_id

    def ingest_directory(
        self,
        db_id: str,
        folder: str,
        suffixes: Optional[List[str]] = None,
        **kwargs
    ) -> List[str]:
        suffixes = suffixes or ['.pdf', '.txt', '.md', '.docx']
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
        upload = os.path.join(base, 'uploads')
        os.makedirs(base, exist_ok=True)
        os.makedirs(upload, exist_ok=True)
        return base, upload

    # -- Milvus Collection 操作 --------------------
    def add_collection(self, name: str, dimension: int) -> None:
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

        from pymilvus import Collection
        collection = Collection(name)
        collection.create_index(
            field_name="vector",
            index_params={
                "index_type": "IVF_FLAT",
                "metric_type": "L2",
                "params": {"nlist": 1024}
            }
        )
        collection.load()

    def get_collection_info(self, name: str) -> Dict[str, Any]:
        try:
            info = self.client.describe_collection(name)
            info.update(self.client.get_collection_stats(name))
            return info
        except MilvusException as e:
            return {'name': name, 'error': str(e)}

    def _insert_vectors(
            self,
            collection_name: str,
            file_id: str,
            docs: List[str],
            chunk_infos: List[Dict[str, Any]]
    ) -> Any:
        if not self.client.has_collection(collection_name):
            raise ValueError("Collection不存在")

        vecs = self.embed_model.batch_encode(docs)

        entities = []
        for idx, v in enumerate(vecs):
            meta = chunk_infos[idx]
            meta["file_id"] = file_id
            meta["text"] = docs[idx]
            vector_id = f"{file_id}_{idx}"
            entities.append({
                "id": vector_id,
                "vector": v,
                "file_id": meta["file_id"],
                "text": meta["text"]
            })

        return self.client.insert(collection_name=collection_name, data=entities)

    # -- 检索 --------------------------------------------------------------
    def search(
            self,
            query: str,
            db_id: str,
            distance_threshold: Optional[float] = None,
            rerank: bool = True,
            top_k: Optional[int] = None
    ) -> Dict[str, Any]:
        dt = distance_threshold or self.default_distance_threshold
        tk = top_k or self.top_k

        #  向量化查询
        vector = self.embed_model.batch_encode([query])[0]

        #  Milvus 检索
        raw_res = self.client.search(
            db_id,
            [vector],
            limit=self.default_max_query_count,
            output_fields=["text", "file_id"]
        )

        # 转成纯Python可序列化结构
        hits: List[Dict[str, Any]] = raw_res[0]
        results: List[Dict[str, Any]] = []
        for h in hits:
            results.append(
                {
                    "entity": {
                        "text": h.get("text", ""),
                        "file_id": h.get("file_id"),
                        "id": h.get("id")  # 其它字段按需保留
                    },
                    "distance": h.get("score", h.get("distance", 0.0))
                }
            )

        # 距离阈值过滤
        filtered = [r for r in results if r["distance"] < dt]

        # 重排序
        if rerank and self.reranker and filtered:
            texts = [r["entity"]["text"] for r in filtered]
            scores = self.reranker.compute_score([query, texts], normalize=False)
            for r, s in zip(filtered, scores):
                r["rerank_score"] = float(s)  # 转 float，保证可 JSON
            filtered = [
                r for r in filtered
                if r["rerank_score"] > self.default_rerank_threshold
            ]
            filtered.sort(key=lambda x: x["rerank_score"], reverse=True)

        return {
            "results": filtered[:tk],
            "all_results": results
        }

    def restart(self):
        self._load_embedding_model(None)
        self._connect_milvus(None)
