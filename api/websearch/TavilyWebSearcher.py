import logging
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import asyncio

from tavily import TavilyClient
from api.websearch.milvus_service import MilvusService
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

logger = logging.getLogger("IndustrialWebSearcher")
logger.setLevel(logging.INFO)
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
from configs.settings import *


class MissingAPIKeyError(Exception):
    pass


class TavilySearchError(Exception):
    pass


def _validate_api_key(api_key: Optional[str]) -> str:
    if not api_key:
        raise MissingAPIKeyError("环境变量未设置。")
    return api_key


def _extract_search_results(raw_results: Dict, max_results: int) -> List[Dict]:
    if not raw_results or "results" not in raw_results:
        logger.warning("Tavilyapi返回内容为空或格式异常。")
        return []
    extracted = []
    for item in raw_results['results'][:max_results]:
        extracted.append({
            'title': item.get('title', ''),
            'content': item.get('content', ''),
            'url': item.get('url', ''),
            'score': item.get('score', 0)
        })
    return extracted


class IndustrialWebSearcher:
    def __init__(self, api_key: Optional[str] = None, cache_ttl_minutes: int = 10, milvus_collection: str = "test"):
        self.api_key = _validate_api_key(api_key or os.getenv("TAVILY_API_KEY"))
        self.client = TavilyClient(self.api_key)
        self.cache_ttl = timedelta(minutes=cache_ttl_minutes)
        self._search_cache: Dict[str, Dict] = {}
        # 初始化 MilvusService（负责插入向量和语义搜索）
        self.milvus = MilvusService(collection_name=milvus_collection, embedding_model=EMBEDDING_MODEL,
                                    overwrite=True)
        logger.info("IndustrialWebSearcher 实例已创建。")

    def _is_cache_valid(self, query: str) -> bool:
        if query not in self._search_cache:
            return False
        cache_entry = self._search_cache[query]
        if 'expires_at' not in cache_entry:
            return False
        return datetime.now() < cache_entry['expires_at']

    def _cache_results(self, query: str, results: List[Dict]) -> None:
        self._search_cache[query] = {
            "results": results,
            "expires_at": datetime.now() + self.cache_ttl
        }

    def perform_search(self, query: str, max_results: int = 3, search_depth: str = "basic") -> List[Dict]:
        """
        调用 Tavily API 获取搜索结果，并缓存结果。之后返回原始结果列表。
        """
        if not query.strip():
            logger.warning("查询字符串为空，将返回空结果。")
            return []
        if max_results <= 0:
            logger.warning("max_results小于等于0，返回空列表。")
            return []
        if self._is_cache_valid(query):
            logger.info(f"缓存命中：'{query}'")
            return self._search_cache[query]["results"]

        start_time = time.time()
        try:
            logger.info(f"开始搜索: '{query}', 深度: {search_depth}, 期望结果数: {max_results}")
            raw_response = self.client.search(
                query=query,
                search_depth=search_depth,
                max_results=max_results
            )
        except Exception as e:
            logger.error(f"Tavily搜索时出现异常：{e}")
            raise TavilySearchError(f"Tavily 搜索异常：{e}")
        results = _extract_search_results(raw_response, max_results)
        self._cache_results(query, results)
        elapsed = time.time() - start_time
        logger.info(f"搜索完成，耗时: {elapsed:.2f} 秒，返回 {len(results)} 条结果。")
        return results

    def format_results(self, results: List[Dict]) -> str:
        """将搜索结果格式化为可读文本"""
        if not results:
            return "没有找到相关的网络搜索结果。"
        lines = ["以下是网络搜索结果：\n"]
        for i, item in enumerate(results, 1):
            lines.append(f"{i}. 标题: {item['title']}")
            lines.append(f"概要: {item['content']}")
            lines.append(f"链接: {item['url']}")
            lines.append(f"分数: {item['score']}\n")
        return "\n".join(lines)

    def clear_cache(self):
        self._search_cache.clear()
        logger.info("搜索缓存已清空。")

    def insert_and_rerank(self, query: str, results: List[Dict], rag_top_k: int = 3) -> List[Dict]:
        """
        利用 MilvusService 将搜索结果插入向量库，并基于向量相似性进行重排序与去重。
        返回语义上最相关的结果列表。
        """
        # 转换为Document格式
        documents = []
        for item in results:
            # page_content摘要或内容，metadata 中记录标题和URL等
            doc = type("Document", (), {})()  # 创建一个简单的对象用于存储属性
            doc.page_content = item['content']
            doc.metadata = {"title": item['title'], "url": item['url'], "snippet": item['content']}
            documents.append(doc)

        # 插入文档到Milvus
        self.milvus.insert_documents(documents)
        # 利用Milvus进行相似性查询，得到重排序后的最相关结果
        retrieved = self.milvus.similarity_search(query, k=rag_top_k)
        # 从检索结果中提取关键信息返回
        reranked = []
        seen = set()
        for doc in retrieved:
            content = doc.page_content
            if content not in seen:
                seen.add(content)
                reranked.append({
                    "title": doc.metadata.get("title", ""),
                    "content": content,
                    "url": doc.metadata.get("url", "")
                })
        return reranked


class IndustrialWebSearcherLLM:
    """
    将 IndustrialWebSearcher与 LLM整合，根据搜索结果生成回答。
    加入 MilvusService 后，会利用向量搜索对搜索结果进行去重和重排序。
    """

    def __init__(
            self,
            api_key: Optional[str] = None,
            cache_ttl_minutes: int = 10,
            milvus_collection: str = "test",
            rag_top_k: int = 3,
            llm_model: str = MODEL_NAME,
            openai_base_url: str = MODEL_API_BASE,
            openai_api_key: str = MODEL_API_KEY,
            embedding_model: str = EMBEDDING_MODEL,
    ):
        self.searcher = IndustrialWebSearcher(api_key=api_key, cache_ttl_minutes=cache_ttl_minutes,
                                              milvus_collection=milvus_collection)
        self.rag_top_k = rag_top_k
        # 提示模板
        self.prompt_templates = {
            "with_context": PromptTemplate(
                template=(
                    "<指令> 根据你实时联网检索到的信息，专业回答用户提出的问题。如果无法从中得到答案，请回复“根据检索到的信息无法回答该问题”。"
                    "同时，如果存在历史对话信息，请结合历史对话提供完整回答，答案请使用中文。\n"
                    "<联网检索到的信息>{context}</联网检索到的信息>\n"
                    "<问题>{question}</问题>\n"
                ),
                input_variables=["context", "question"]
            ),
            "without_context": PromptTemplate(
                template="请你回答我的问题:\n{question}\n\n",
                input_variables=["question"]
            )
        }
        self.llm = ChatOpenAI(
            model=llm_model,
            base_url=openai_base_url,
            api_key=openai_api_key,
        )

    def _combine_results(self, raw_results: List[Dict], reranked: List[Dict]) -> str:
        """
        合并原始搜索结果（例如摘要）与Milvus 重排序后的结果，生成最终的上下文字符串，
        并通过set去重。
        """
        combined_snippets = []
        # 提取原始结果摘要
        for item in raw_results:
            combined_snippets.append(f"标题: {item['title']}\n摘要: {item['content']}\n链接: {item['url']}")
        # 提取重排序后的结果
        for item in reranked:
            combined_snippets.append(f"标题: {item['title']}\n摘要: {item['content']}\n链接: {item['url']}")
        # 利用set去重后，用两个换行符拼接
        unique_contexts = list(set(combined_snippets))
        return "\n\n".join(unique_contexts)

    async def generate_answer(self, question: str, context: str) -> str:
        """
        根据问题和上下文使用 LLM 生成回答。
        """
        template = self.prompt_templates["with_context"] if context else self.prompt_templates["without_context"]
        input_dict = {"question": question, "context": context}
        chain = template | self.llm
        response = await chain.ainvoke(input_dict)
        return response.content

    async def search_and_generate(self, query: str, max_results: int = 3, search_depth: str = "basic") -> str:
        """
        先利用IndustrialWebSearcher获取原始结果，再利用Milvus进行去重重排序，
        最后基于合并的上下文使用 LLM 生成回复。
        """
        raw_results = self.searcher.perform_search(query, max_results=max_results, search_depth=search_depth)
        reranked_results = self.searcher.insert_and_rerank(query, raw_results, rag_top_k=self.rag_top_k)
        context = self._combine_results(raw_results, reranked_results)
        answer = await self.generate_answer(question=query, context=context)
        return answer


if __name__ == '__main__':
    async def main():
        searcher_llm = IndustrialWebSearcherLLM(
            api_key=TAVILY_API_KEY,
            cache_ttl_minutes=10,
            milvus_collection="test2",
            rag_top_k=3,
            llm_model=MODEL_NAME,
            openai_api_key=MODEL_API_KEY,
            openai_base_url=MODEL_API_BASE
        )
        query_str = "皮卡丘进化是什么？"
        logger.info(f"正在执行搜索查询: {query_str}")
        answer = await searcher_llm.search_and_generate(query_str, max_results=3, search_depth="basic")
        print("===== 模型回复 =====")
        print(answer)

        # # 打印格式化的原始搜索结果
        # formatted_results = searcher_llm.searcher.format_results(
        #     searcher_llm.searcher.perform_search(query_str, max_results=3, search_depth="basic")
        # )
        # print("\n=== 格式化后的搜索结果 ===")
        # print(formatted_results)


    asyncio.run(main())
