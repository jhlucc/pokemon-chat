import os

from src.knowledge.core.indexing import chunk_file
from src.knowledge.core.Milvus import MilvusStorage
from src.models.embedding import get_embedding_model
from src.utils.logger import get_logger

_log = get_logger(__name__)


# 知识导入到向量数据库
class DocumentIngestor:
    def __init__(
        self,
        milvus_config: dict,
        embedding_config: dict,
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
        ocr_enabled: bool = False,
        ocr_det_threshold: float = 0.3,
    ):
        """
        文件 -> Chunk -> Embedding -> 存入 Milvus 的流程
        Args:
            milvus_config: dict, 用于初始化 MilvusStorage
            embedding_config: dict, 用于初始化 Embedding 模型
            chunk_size: 分块大小
            chunk_overlap: 分块重叠
            ocr_enabled: 是否对 PDF 启用 OCR
            ocr_det_threshold: OCR 检测阈值
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.ocr_enabled = ocr_enabled
        self.ocr_det_threshold = ocr_det_threshold

        # 1) 加载 embedder
        self.embedder = get_embedding_model(embedding_config)
        # 2) 初始化Milvus
        self.store = MilvusStorage(**milvus_config)

    def ingest_single_file(self, file_path: str):
        """
        处理单文件 -> chunk -> embedding -> 写入 Milvus
        """
        _log.info(f"Processing file: {file_path}")
        # 步骤 1: chunk
        chunks = chunk_file(
            file_path=file_path,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            do_ocr=self.ocr_enabled,
            ocr_det_threshold=self.ocr_det_threshold,
        )

        # 步骤 2: embedding
        texts = [doc.page_content for doc in chunks]

        embeddings = self.embedder.batch_encode(texts)

        # 步骤 3: 把embedding放到 doc.metadata["embedding"]
        for doc, emb in zip(chunks, embeddings):
            doc.metadata["embedding"] = self._normalize_embedding(emb)
        # 步骤 4: 调用 MilvusStorage.insert
        self.store.insert(chunks)
        _log.info(f"File ingested into Milvus: {file_path}")

    @staticmethod
    def _normalize_embedding(emb) -> list[float]:
        if hasattr(emb, "tolist"):
            emb = emb.tolist()
        elif isinstance(emb, str):
            raise ValueError(f"Embedding 不应为字符串类型: {emb[:100]}...")
        if not isinstance(emb, list):
            raise TypeError(f"Embedding 类型错误，必须是 list[float] {type(emb)}")
        return emb

    def ingest_directory(self, directory_path: str, suffixes: list[str] | None = None):
        """
        批量处理一个文件夹
        """
        suffixes = suffixes or [".pdf", ".docx", ".txt", ".md"]
        _log.info(f"Scanning directory: {directory_path}")

        for root, _dirs, files in os.walk(directory_path):
            for file in files:
                if any(file.lower().endswith(ext) for ext in suffixes):
                    file_path = os.path.join(root, file)
                    self.ingest_single_file(file_path)

    def close(self):
        """关闭 Milvus 连接"""
        self.store.close()
        _log.info("Milvus connection closed.")


if __name__ == "__main__":
    # 示例配置
    milvus_config = {
        "collection_name": "documents_collection1",
        "overwrite": True,
        "dim": 1024,
        "host": "localhost",
        "port": "19530",
    }

    embedding_config = {"enable_knowledge_base": True, "embed_model": "openai/bge-m3-pro"}

    ingestor = DocumentIngestor(
        milvus_config=milvus_config,
        embedding_config=embedding_config,
        chunk_size=500,
        chunk_overlap=100,
        ocr_enabled=True,
    )

    ingestor.ingest_single_file("/data/Langagent/deepdoc/data/random_data.csv")
    # 或处理一个目录
    # ingestor.ingest_directory("/data/docs")

    ingestor.close()
