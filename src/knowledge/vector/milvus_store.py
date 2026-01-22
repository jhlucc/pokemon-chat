import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
from typing import List, Optional
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from pymilvus import (
    connections, FieldSchema, CollectionSchema,
    DataType, Collection, utility
)
from src.core.settings import settings
from src.models.reranker_model import *


class MilvusService:
    def __init__(
            self,
            collection_name: str = "test",
            dim: int = 1024,
            host: str = "localhost",
            port: str = "19530",
            overwrite: bool = False,
            openai_base_url: str = None,
            openai_api_key: str = None,
            embedding_model: str = None,
            reranker_key: str = "siliconflow/bge-reranker-v2-m3",
            reranker_local_path: str = None,
            reranker_model: str = 'BAAI/bge-reranker-v2-m3',
    ):
        # Apply defaults from settings
        openai_base_url = openai_base_url or settings.llm.api_base
        openai_api_key = openai_api_key or settings.llm.api_key
        embedding_model = embedding_model or settings.embedding.model_name
        reranker_local_path = reranker_local_path or str(settings.paths.model_reranker_path)
        """
        初始化 Milvus 向量存储
        参数:
            collection_name: 集合名称
            dim: 向量维度
            host: Milvus 服务器地址
            port: Milvus 端口
            overwrite: 是否覆盖现有集合
            openai_api_key: OpenAI API 密钥
            embedding_model: OpenAI 嵌入模型名称
        """
        # 连接 Milvus
        connections.connect(host=host, port=port)

        self.collection_name = collection_name
        self.dim = dim

        self.embedder = OpenAIEmbeddings(
            model=embedding_model,
            openai_api_base=openai_base_url,
            openai_api_key=openai_api_key,
            chunk_size=32
        )
        self.reranker = RerankerWrapper(
            reranker_key=reranker_key,
            model_name=reranker_model,
            local_path=reranker_local_path,
            device="cpu"
        )
        # 定义集合结构
        self.fields = [
            FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
            FieldSchema(name="metadata", dtype=DataType.JSON),
            FieldSchema(name="text_length", dtype=DataType.INT64)
        ]

        self.schema = CollectionSchema(
            fields=self.fields,
            description="LangChain Documents with OpenAI Embeddings",
            enable_dynamic_field=True
        )

        # 检查集合是否存在
        if utility.has_collection(collection_name):
            if overwrite:
                utility.drop_collection(collection_name)
                print(f"已覆盖现有集合: {collection_name}")
                self.collection = self._create_collection()
            else:
                self.collection = Collection(collection_name)
                print(f"已加载现有集合: {collection_name}")
        else:
            self.collection = self._create_collection()

        # 创建索引（如果不存在）
        if not self.collection.has_index():
            self._create_index()

    def _create_collection(self) -> Collection:
        """创建新集合"""
        print(f"创建新集合: {self.collection_name}")
        return Collection(
            name=self.collection_name,
            schema=self.schema,
            consistency_level="Strong"
        )

    def _create_index(self, index_params: Optional[dict] = None):
        """创建向量索引 (优化用于 OpenAI 嵌入)"""
        default_index = {
            "index_type": "IVF_FLAT",
            "metric_type": "COSINE",  # OpenAI 推荐使用余弦相似度
            "params": {"nlist": 16}  # 较大值提高搜索精度
        }
        params = index_params or default_index

        print(f"创建索引: {params}")
        self.collection.create_index(
            field_name="embedding",
            index_params=params
        )
        self.collection.load()

    def insert_documents(self, documents: List[Document], batch_size: int = 32):
        """
        插入 LangChain 文档列表 (自动生成 OpenAI 嵌入)
        参数:
            documents: LangChain Document 对象列表
            batch_size: 分批处理大小 (避免API限流)
        """
        total_docs = len(documents)
        print(f"开始插入 {total_docs} 个文档 (分批大小: {batch_size})")

        for i in range(0, total_docs, batch_size):
            batch_docs = documents[i:i + batch_size]
            # 生成嵌入
            texts = [doc.page_content for doc in batch_docs]
            embeddings = self.embedder.embed_documents(texts)
            text_length = [len(doc.page_content) for doc in batch_docs]
            # 准备数据
            metadatas = [doc.metadata for doc in batch_docs]

            # 插入批次数据
            entities = [
                texts,
                embeddings,
                metadatas,
                text_length
            ]

            try:
                self.collection.insert(entities)
                print(f"已插入 {min(i + batch_size, total_docs)}/{total_docs} 文档")
            except Exception as e:
                print(f"插入失败 (文档 {i}-{i + batch_size}): {str(e)}")
                raise

        self.collection.flush()
        print(f"文档插入完成! 总计: {self.collection.num_entities} 个文档")

    def similarity_search(
            self,
            query: str,
            k: int = 3,
            rerank: bool = True,
            **search_kwargs
    ) -> List[Document]:
        # 生成查询嵌入
        query_embedding = self.embedder.embed_query(query)

        # 构建长度过滤表达式
        length_filter = "text_length > 50"

        # 搜索参数
        search_params = {
            "metric_type": "COSINE",
            "params": {"nprobe": 8, "ef": 64},
            **search_kwargs
        }

        # 执行搜索
        search_result = self.collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=k * 5 if rerank else k,  # 扩大检索量确保结果充足
            expr=length_filter,  # 关键修改：数据库层过滤
            output_fields=["text", "metadata"],
            consistency_level="Strong"
        )

        candidates = []
        for hits in search_result:
            for hit in hits:
                text = getattr(hit.entity, "text", "")
                metadata = getattr(hit.entity, "metadata", {})

                doc = Document(
                    page_content=text,
                    metadata={
                        **metadata,
                        "distance": hit.distance,
                        "raw_score": 1 - hit.distance  # 存储原始相似度分数
                    }
                )
                candidates.append(doc)

        # 如果没有启用rerank，直接返回前k个结果
        if not rerank or len(candidates) <= k:
            return candidates[:k]
        # 执行rerank
        return self.rerank_documents(query, candidates, k)

    def rerank_documents(self, query: str, candidates: List[Document], k: int) -> List[Document]:
        print("🎯 正在调用 reranker 对候选文档重新排序...")
        docs = [doc.page_content for doc in candidates]
        scores = self.reranker.run(query, docs, normalize=True)

        for doc, score in zip(candidates, scores):
            doc.metadata["rerank_score"] = float(score)

        candidates.sort(key=lambda x: x.metadata["rerank_score"], reverse=True)
        return candidates[:k]

    def hybrid_search(
            self,
            query: str,
            k: int = 5,
            filter_expr: Optional[str] = None,
            keyword_weight: float = 0.3,
            vector_weight: float = 0.7
    ) -> List[Document]:
        """
        混合搜索 (结合关键词和向量相似度)
        
        参数:
            query: 查询文本
            k: 返回结果数量
            filter_expr: 元数据过滤表达式
            keyword_weight: 关键词分数权重
            vector_weight: 向量分数权重
        """
        # 向量搜索
        vector_results = self.similarity_search(query, k * 2)

        # 关键词搜索 (简单实现)
        query_lower = query.lower()

        def keyword_score(text):
            words = set(query_lower.split())
            text_words = set(text.lower().split())
            return len(words & text_words) / len(words) if words else 0

        # 合并结果
        scored_docs = []
        for doc in vector_results:
            vector_score = 1 - doc.metadata.get("distance", 0)  # 转换距离为相似度
            text_score = keyword_score(doc.page_content)
            combined_score = (vector_score * vector_weight) + (text_score * keyword_weight)
            scored_docs.append((combined_score, doc))

        # 按综合评分排序
        scored_docs.sort(reverse=True, key=lambda x: x[0])
        return [doc for _, doc in scored_docs[:k]]

    def clear_collection(self):
        """清空集合数据"""
        utility.drop_collection(self.collection_name)
        self.collection = self._create_collection()
        self._create_index()
        print("集合已清空")

    def close(self):
        """关闭连接"""
        connections.disconnect(alias="one")
        print("Milvus 连接已关闭")


# 使用示例
if __name__ == "__main__":
    # 初始化
    vector_store = MilvusService(
        collection_name="test",
        overwrite=True,
        # embedding_model="bge-m3-pro",
        # reranker_key="siliconflow/bge-reranker-v2-m3",  # 或 "local/bge-reranker-v2-m3"
        # reranker_model="BAAI/bge-reranker-v2-m3"  # 或本地路径
    )

    # 示例文档
    docs = [
        Document(
            page_content='#\n\n  * 电视动画系列\n  * 电影\n  * 商品\n  * 游戏\n  * 集换式卡牌游戏\n  * 宝可梦图鉴\n\nClose\n\n宝可梦图鉴\n\n0024 阿柏怪\n\n0025\n\n皮卡丘\n\n皮卡丘 0025\n\n属性\n\n电\n\n弱点\n\n地面\n\n身高 0.4 m\n\n分类 鼠宝可梦\n\n体重 6.0 kg\n\n性别 /\n\n特性 静电\n\n特性 静电 身上带有静电，有时会让接触到的对手麻痹。\n\n图鉴版本\n\n两颊上有储存电力的囊袋。一旦生气就会把储存的电力一口气释放出来。\n\n据说当好几只聚在一起时，那里就会凝聚强烈的电力，还可能会落下闪电。\n\n双颊有囊，用以积蓄电力。栖息在森林，性情聪慧，会以电击烧灼坚硬的树果食用。\n\n能力\n\nHP\n\n攻击\n\n防御\n\n特攻\n\n特防\n\n速度\n\n样子\n\n0025 皮卡丘\n\n电\n\n0025 皮卡丘 超极巨化\n\n电\n\n进化\n\n0172\n\n皮丘\n\n电\n\n0025\n\n皮卡丘\n\n电\n\n0026\n\n雷丘\n\n电\n\n0026\n\n雷丘\n\n阿罗拉的样子\n\n电\n\n超能力\n\n  * \n\n返回Pokédex',
            metadata={'uuid': 'e8926baa84557853ea8df3288161e77c',
                      'title': '皮卡丘| 宝可梦图鉴| The official Pokémon Website in China',
                      'snippet': '皮卡丘. 皮卡丘 0025. 属性. 电. 弱点. 地面. 身高 0.4 m. 分类 鼠宝可梦. 体重 6.0 kg. 性别 /. 特性 ... 0025 皮卡丘 超极巨化. 电. 进化. 0172. 皮丘. 电 · 0025. 皮卡丘.',
                      'link': 'https://www.pokemon.cn/play/pokedex/0025', 'score': 0.04201680672268904}
        ),
        Document(
            page_content='#\n\n  * 電視動畫系列\n  * 電影\n  * 商品\n  * 應用程式\n  * 遊戲\n  * 活動\n  * 卡牌遊戲\n  * 寶可夢圖鑑\n\nClose\n\n寶可夢圖鑑\n\n0024 阿柏怪\n\n0025\n\n皮卡丘\n\n皮卡丘 0025\n\n屬性\n\n電\n\n弱點\n\n地面\n\n身高 0.4 m\n\n分類 鼠寶可夢\n\n體重 6.0 kg\n\n性別 /\n\n特性 靜電\n\n特性 靜電 身上帶有靜電，有時會令接觸到的對手麻痺。\n\n圖鑑版本\n\n雙頰上有儲存電力的囊袋。一旦生氣就會把儲存的電力一口氣釋放出來。\n\n據說同一處有好幾隻的時候，那裡就會凝集起強烈的電力，還可能造成閃電落於該處。\n\n雙頰有囊，用以積蓄電力。棲息在森林，性情聰慧，會以電擊燒灼堅硬的樹果食用。\n\n能力\n\nHP\n\n攻擊\n\n防禦\n\n特攻\n\n特防\n\n速度\n\n樣子\n\n0025 皮卡丘\n\n電\n\n0025 皮卡丘 超極巨化\n\n電\n\n進化\n\n0172\n\n皮丘\n\n電\n\n0025\n\n皮卡丘\n\n電\n\n0026\n\n雷丘\n\n電\n\n0026\n\n雷丘\n\n阿羅拉的樣子\n\n電\n\n超能力\n\n  *   * \n\n返回Pokédex',
            metadata={'uuid': '27998f19ce9eeee669d077a9546a9c78',
                      'title': '皮卡丘| 寶可夢圖鑑| The official Pokémon Website in Taiwan',
                      'snippet': '皮卡丘. 皮卡丘 0025. 屬性. 電. 弱點. 地面. 身高 0.4 m. 分類 鼠寶可夢. 體重 6.0 kg. 性別 /. 特性 ... 0025 皮卡丘 超極巨化. 電. 進化. 0172. 皮丘. 電 · 0025. 皮卡丘.',
                      'link': 'https://tw.portal-pokemon.com/play/pokedex/0025', 'score': 0.03361344537815125}
        ),
        Document(
            page_content="皮卡丘的进化是雷丘，它们都是电属性宝可梦。雷丘比皮卡丘体型更大，攻击力也更强。",
            metadata={"uuid": "test-001", "title": "皮卡丘进化介绍"}
        ),
        Document(
            page_content="皮丘是皮卡丘的退化形态，当与训练师建立足够的亲密关系后，皮丘会进化成皮卡丘。",
            metadata={"uuid": "test-002", "title": "皮丘与皮卡丘"}
        ),
        Document(
            page_content="哈利波特是一部非常著名的魔法小说，与宝可梦无关。",
            metadata={"uuid": "test-003", "title": "哈利波特"}
        ),
        Document(
            page_content="天气晴朗，适合郊游或者在公园散步。",
            metadata={"uuid": "test-004", "title": "天气与出行"}
        ),
        Document(
            page_content="雷丘还有阿罗拉形态，是皮卡丘的特殊进化之一。",
            metadata={"uuid": "test-005", "title": "雷丘阿罗拉形态"}
        ),
    ]

    # 插入文档 (自动生成嵌入)
    vector_store.insert_documents(docs)

    # 相似性搜索
    query = "皮卡丘的进化是什么？"
    print(f"\nQuery: {query}")
    results = vector_store.similarity_search(query, k=1)

    for i, doc in enumerate(results):
        print(f"\nResult {i + 1}:")
        print(f"Content: {doc.page_content}")
        print(f"Metadata: {doc.metadata}")
        print(f"Similarity: {1 - doc.metadata.get('distance', 0):.3f}")

    # 混合搜索示例
    # print("\nHybrid Search Results:")
    # hybrid_results = vector_store.hybrid_search(query, k=2)
    # for doc in hybrid_results:
    #     print(f"- {doc.page_content[:60]}...")

    # 清理
    vector_store.close()
