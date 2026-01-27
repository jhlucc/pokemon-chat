"""
Query Decomposition for Complex Questions

Breaks down complex queries into simpler sub-queries,
retrieves for each, and merges results.
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from src.core.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SubQueries(BaseModel):
    """Decomposed sub-queries."""

    queries: list[str] = Field(description="List of simpler sub-queries that together answer the original question")
    reasoning: str = Field(description="Brief explanation of why these sub-queries were chosen")


DECOMPOSITION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """你是宝可梦专家级查询分解器。
将复杂的宝可梦问题分解为2-4个可独立回答的子问题。

分解规则：
1. 每个子问题应该是自包含的，可以独立检索回答
2. 子问题应覆盖原问题的不同方面
3. 简单问题不需要分解，直接返回原问题
4. 关注可通过知识库检索回答的事实性子问题

宝可梦专属示例：
- "皮卡丘和小火龙谁更强" -> ["皮卡丘的种族值是多少?", "小火龙的种族值是多少?", "电系和火系的属性相克关系?"]
- "如何用喷火龙打败水系道馆" -> ["喷火龙可以学习哪些非火系技能?", "喷火龙的隐藏特性是什么?", "对水系有效的草系和电系技能有哪些?"]
- "皮卡丘有几个进化形态" -> ["皮卡丘有几个进化形态"] (简单问题，无需分解)
- "卡比兽和快龙哪个适合做队伍主力" -> ["卡比兽的种族值和特性?", "快龙的种族值和特性?", "两者各自的优势对战场景?"]""",
        ),
        ("human", "问题: {query}\n\n请分解为子查询:"),
    ]
)


class QueryDecomposer:
    """
    Decomposes complex queries into simpler sub-queries.
    """

    def __init__(self):
        self.llm = ChatOpenAI(
            model=settings.llm.model_name, api_key=settings.llm.api_key, base_url=settings.llm.api_base, temperature=0
        )
        self.chain = DECOMPOSITION_PROMPT | self.llm.with_structured_output(SubQueries)

    def decompose(self, query: str) -> list[str]:
        """
        Decompose a query into sub-queries.

        Args:
            query: Original complex query

        Returns:
            List of sub-queries (may be just the original if simple)
        """
        try:
            result = self.chain.invoke({"query": query})
            logger.info(f"Query decomposed: {query[:50]}... -> {len(result.queries)} sub-queries")
            return result.queries
        except Exception as e:
            logger.warning(f"Query decomposition failed: {e}, using original")
            return [query]

    def is_complex(self, query: str) -> bool:
        """
        Heuristic to check if a query is complex enough to decompose.
        """
        # Simple heuristics
        complex_indicators = [
            " and ",
            " vs ",
            " versus ",
            " compare ",
            " difference ",
            " between ",
            " stronger ",
            " better ",
            " which ",
        ]
        query_lower = query.lower()
        return any(indicator in query_lower for indicator in complex_indicators)


# Global instance
_decomposer: QueryDecomposer = None


def get_query_decomposer() -> QueryDecomposer:
    global _decomposer
    if _decomposer is None:
        _decomposer = QueryDecomposer()
    return _decomposer
