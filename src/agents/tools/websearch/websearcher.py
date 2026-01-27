import asyncio
import concurrent.futures  # ← 新增
import logging
import os
from abc import ABC, abstractmethod
from typing import Any

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from src.core.settings import settings
from src.models.schemas import Source

# -------------------- 全局超时时长（秒） --------------------
SEARCH_TIMEOUT = 10  # 你可以按需改成 5、15 等
# -----------------------------------------------------------

logger = logging.getLogger("WebSearcher")
logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# 抽象基类
# ---------------------------------------------------------------------------


class BaseWebSearcher(ABC):
    """所有搜索器统一接口：同步 search(query) -> List[Source]"""

    @abstractmethod
    def search(self, query: str, top_k: int = 5) -> list[Source]:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Tavily
# ---------------------------------------------------------------------------


class TavilyBasicSearcher(BaseWebSearcher):
    """使用 Tavily API 进行搜索（完全同步实现，已加超时）"""

    def __init__(self, api_key: str | None = None):
        from tavily import TavilyClient

        self.api_key = api_key or os.getenv("TAVILY_API_KEY")
        if not self.api_key:
            raise ValueError("Tavily API Key 未提供!!!")
        self.client = TavilyClient(self.api_key)

    def _safe_call(self, *args, **kwargs):
        """在线程池里调用，便于加超时"""
        return self.client.search(*args, **kwargs)

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        logger.info(f"[TavilyBasicSearcher] Searching for: {query} (top_k={top_k})")
        if not query.strip():
            return []

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(self._safe_call, query=query, max_results=top_k, search_depth="basic")
                raw = future.result(timeout=SEARCH_TIMEOUT)
        except concurrent.futures.TimeoutError:
            logger.warning(f"Tavily 搜索超时 (> {SEARCH_TIMEOUT}s)")
            return []
        except Exception as e:
            logger.error(f"Tavily 搜索异常: {e}")
            return []

        if "results" not in raw:
            logger.warning("Tavily 响应中未找到 results 字段")
            return []

        return [
            Source(
                title=item.get("title", ""),
                content_snippet=item.get("content", ""),
                url=item.get("url", ""),
                score=item.get("score", 0),
            )
            for item in raw["results"][:top_k]
        ]


# ---------------------------------------------------------------------------
# Lite 基础搜索  ——  用 async 搜索工具，但对外提供同步接口
# ---------------------------------------------------------------------------


class LiteBaseSearcher(BaseWebSearcher):
    def __init__(self, some_config: Any | None = None):
        self.llm = ChatOpenAI(
            model=settings.llm.model_name, base_url=settings.llm.api_base, api_key=settings.llm.api_key, temperature=0.0
        )

    def _check_safety(self, query: str) -> dict[str, Any]:
        """检查查询是否安全/相关"""
        prompt = ChatPromptTemplate.from_template("""
        你是 Pokemon 搜索守门人。你的任务是判断用户的搜索查询是否与 "Pokemon (宝可梦/口袋妖怪)"、"动画/游戏" 相关。

        判断规则：
        1. 如果包含宝可梦名称、角色、招式、地点等，返回 "pass"。
        2. 如果是日常问候（你好、早上好等），返回 "pass"。
        3. 如果是完全无关的话题（如：政治、股票、其他动漫、编程问题等），返回 "block"。

        请输出 JSON 格式:
        {{
            "status": "pass" 或 "block",
            "reason": "原因"
        }}

        用户查询: {query}
        """)

        chain = prompt | self.llm | JsonOutputParser()
        try:
            return chain.invoke({"query": query})
        except Exception as e:
            logger.error(f"Guardrail check failed: {e}")
            return {"status": "pass", "reason": "check_failed"}

    # ---------- 内部工具 ----------

    def _run_sync(self, coro):
        """
        在任意环境下安全执行协程，并允许外层指定超时。
        1. 如果当前线程已有事件循环（如 FastAPI/Uvicorn），用 run_coroutine_threadsafe。
        2. 否则直接 asyncio.run。
        """
        try:
            loop = asyncio.get_running_loop()
            future = asyncio.run_coroutine_threadsafe(coro, loop)
            return future.result(timeout=SEARCH_TIMEOUT)
        except RuntimeError:
            # 当前线程没有事件循环（脚本、子线程等）
            return asyncio.run(asyncio.wait_for(coro, timeout=SEARCH_TIMEOUT))

    # ---------- 对外接口 ----------

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        logger.info(f"[LiteWebSearcher] Searching for: {query} (top_k={top_k})")
        if not query.strip():
            return []

        # [Guardrail] 安全检查
        try:
            # 可以在线程中运行 check 以避免阻塞 async loop (如果外层是 loop)
            # 但这里 _run_sync 已经很复杂了，直接在这里同步调用 LLM (invoke is sync by default unless ainvoke)
            # 考虑到 LiteBaseSearcher 主要在 synchronous context (search is sync method) 使用，直接调用即可。
            check_result = self._check_safety(query)
            if check_result.get("status") == "block":
                logger.warning(f"[Guardrail] 拦截搜索: {query} | Reason: {check_result.get('reason')}")
                return [
                    Source(
                        title="搜索被拒绝",
                        content_snippet=f"抱歉，我无法搜索不是宝可梦主题的内容~ (原因: {check_result.get('reason', '无关内容')})",
                        url="#",
                        score=0.0,
                    )
                ]
        except Exception as e:
            logger.error(f"Guardrail check error: {e}")
            # 出错放行
            pass

        # 异步搜索工具（保持原来的导入路径）
        from src.agents.tools.websearch.utils import search  # async def search(q, k) -> list

        try:
            raw_results = self._run_sync(search(query, top_k))
        except asyncio.TimeoutError:
            logger.warning(f"LiteWebSearcher 搜索超时 (> {SEARCH_TIMEOUT}s)")
            return []
        except Exception as e:
            logger.error(f"LiteWebSearcher 搜索异常: {e}")
            return []

        return [
            Source(title=doc.get("title", ""), content_snippet=doc.get("snippet", ""), url=doc.get("link", ""))
            for doc in raw_results[:top_k]
        ]


# ---------------------------------------------------------------------------
# 简单自测
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    query_text = "皮卡丘进化是什么？"

    try:
        tavily_searcher = TavilyBasicSearcher(api_key=settings.tavily.api_key)
        print("[Tavily] ->", tavily_searcher.search(query_text, top_k=3))
    except Exception as e:
        print(f"[Tavily] 初始化失败: {e}")

    lite_searcher = LiteBaseSearcher()
    print("[Lite]   ->", lite_searcher.search(query_text, top_k=3))
