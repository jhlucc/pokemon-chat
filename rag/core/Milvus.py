import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
import numpy as np
import torch
from typing import List, Optional
from langchain.schema import Document
from pymilvus import (
    connections, FieldSchema, CollectionSchema,
    DataType, Collection, utility
)


class MilvusStorage:
    def __init__(
            self,
            collection_name: str = "default",
            dim: int = 1024,
            host: str = "localhost",
            port: str = "19530",
            overwrite: bool = False
    ):
        """
        一个轻量封装:
        - 只负责把带有 embedding 的 Document 存进 Milvus
        - 不做 embedding
        """
        self.collection_name = collection_name
        self.dim = dim

        connections.connect(alias="default", host=host, port=port)

        self.fields = [
            FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
            FieldSchema(name="metadata", dtype=DataType.JSON),
            FieldSchema(name="text_length", dtype=DataType.INT64),
        ]

        self.schema = CollectionSchema(
            fields=self.fields,
            description="Embedded documents",
            enable_dynamic_field=True
        )

        if utility.has_collection(collection_name):
            if overwrite:
                utility.drop_collection(collection_name)
                print(f"已覆盖并删除原有集合: {collection_name}")
                self.collection = self._create_collection()
            else:
                self.collection = Collection(collection_name)
                print(f"已加载现有集合: {collection_name}")
        else:
            self.collection = self._create_collection()

        if not self.collection.has_index():
            self._create_index()

    def _create_collection(self) -> Collection:
        print(f"创建新集合: {self.collection_name}")
        return Collection(
            name=self.collection_name,
            schema=self.schema,
            consistency_level="Strong"
        )

    def _create_index(self, index_params: Optional[dict] = None):
        default_index = {
            "index_type": "IVF_FLAT",
            "metric_type": "COSINE",
            "params": {"nlist": 16}
        }
        self.collection.create_index(
            field_name="embedding",
            index_params=index_params or default_index
        )
        self.collection.load()

    def insert(self, documents: List[Document], batch_size: int = 32):
        """
        注意：要求 doc.metadata["embedding"] 已经存在且维度 == self.dim
        """
        total_docs = len(documents)
        print(f"开始插入 {total_docs} 个文档...")

        inserted = 0
        for i in range(0, total_docs, batch_size):
            batch = documents[i:i + batch_size]
            texts = []
            embeddings = []
            metas = []
            lengths = []

            for doc in batch:
                emb = doc.metadata.get("embedding", None)
                if emb is None:
                    raise ValueError("Document缺少embedding")
                if isinstance(emb, (np.ndarray, torch.Tensor)):
                    emb = emb.tolist()
                if len(emb) != self.dim:
                    raise ValueError(f"embedding长度({len(emb)})与定义的维度({self.dim})不匹配!")

                clean_meta = dict(doc.metadata)
                clean_meta.pop("embedding", None)

                texts.append(doc.page_content)
                embeddings.append(emb)
                metas.append(clean_meta)
                lengths.append(len(doc.page_content))

                texts.append(doc.page_content)
                embeddings.append(emb)
                metas.append(doc.metadata)
                lengths.append(len(doc.page_content))

            entities = [
                texts,  # text
                embeddings,  # embedding
                metas,  # metadata
                lengths,  # text_length
            ]
            self.collection.insert(entities)
            inserted += len(batch)
            print(f"已插入 {inserted}/{total_docs} 条...")

        self.collection.flush()
        print(f"插入完成, {self.collection.num_entities} 条数据在集合 {self.collection_name} 中.")

    def close(self):
        # Milvus官方SDK: 指定alias (默认 "default")
        connections.disconnect(alias="default")
        print("Milvus 连接已关闭")
