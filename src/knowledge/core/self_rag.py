"""
Self-RAG - Self-Reflective Retrieval Augmented Generation

The model decides:
1. Whether to retrieve (or answer from knowledge)
2. Whether retrieved docs are relevant
3. Whether the generated answer is supported by docs
"""

from typing import Any, Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from src.core.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class RetrievalDecision(BaseModel):
    """Decision on whether to retrieve."""

    should_retrieve: bool = Field(description="Whether retrieval is needed")
    reason: str = Field(description="Reason for the decision")
    query_type: Literal["factual", "creative", "conversational", "complex"] = Field(description="Type of query")


class RelevanceCheck(BaseModel):
    """Check if retrieved docs are relevant."""

    is_relevant: bool = Field(description="Whether docs are relevant to query")
    relevance_score: float = Field(ge=0, le=1, description="Relevance score")
    critique: str = Field(description="Critique of the retrieval")


class SupportCheck(BaseModel):
    """Check if answer is supported by docs."""

    is_supported: bool = Field(description="Whether answer is fully supported")
    support_level: Literal["full", "partial", "none"] = Field(description="Level of support from docs")
    unsupported_claims: str = Field(description="Any claims not supported by docs")


RETRIEVAL_DECISION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """你是宝可梦知识检索决策系统。判断用户的问题是否需要检索宝可梦知识库。

需要检索的情况：
- 询问特定宝可梦的属性、技能、进化、特性等信息
- 询问宝可梦之间的对战克制关系
- 询问训练师、道馆、地点等剧情信息
- 询问游戏机制（如努力值、个体值、孤独攻击等）
- 需要准确数据的问题（种族值、申码、身高体重等）

不需要检索的情况：
- 闲聊/问候（你好、今天天气怎么样）
- 要求编故事或开玩笑
- 询问当前对话的内容
- 简单的常识问题（皮卡丘是什么）

示例：
- "皮卡丘的种族值是多少？" → 需要检索（精确数据）
- "我喜欢皮卡丘！" → 不需要检索（闲聊）
- "火系克制草系吗？" → 需要检索（属性克制）
- "给我编一个关于小智的故事" → 不需要检索（创意内容）""",
        ),
        ("human", "用户问题: {query}\n\n是否需要检索宝可梦知识库？"),
    ]
)

RELEVANCE_CHECK_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """你是宝可梦知识相关性评估员。评估检索到的文档是否与用户问题相关。

相关性评分标准：
- 0.9-1.0: 文档直接回答了问题（如问皮卡丘类型，文档明确说了电系）
- 0.7-0.9: 文档包含相关信息但可能需要推理
- 0.4-0.7: 文档部分相关（如问皮卡丘，文档讲的是雷丘）
- 0.0-0.4: 文档不相关（如问火系，文档讲的是水系）

要严格但合理。如果文档讲的是相关宝可梦或者相关机制，也算有一定相关性。""",
        ),
        (
            "human",
            """用户问题: {query}

检索到的宝可梦知识:
{documents}

这些文档对回答问题有帮助吗？""",
        ),
    ]
)

SUPPORT_CHECK_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """你是宝可梦知识事实核查员。检查生成的回答是否有文档支持。

检查要点：
- 宝可梦名称、属性、类型是否精确
- 数值数据（种族值、HP、攻击等）是否与文档一致
- 进化链、学习技能等是否有据可查
- 是否有文档中未提及的主张（幻觉）

支持级别：
- full: 所有声明都有文档支持
- partial: 部分声明有支持，部分来自常识
- none: 主要声明没有文档支持""",
        ),
        (
            "human",
            """用户问题: {query}

参考文档:
{documents}

生成的回答:
{answer}

这个回答是否有文档支持？请指出任何没有支持的声明。""",
        ),
    ]
)


class SelfRAG:
    """
    Self-RAG implementation with reflection tokens.

    Provides adaptive retrieval and answer verification.
    """

    def __init__(self):
        self.llm = ChatOpenAI(
            model=settings.llm.model_name, api_key=settings.llm.api_key, base_url=settings.llm.api_base, temperature=0
        )

        self.retrieval_chain = RETRIEVAL_DECISION_PROMPT | self.llm.with_structured_output(RetrievalDecision)
        self.relevance_chain = RELEVANCE_CHECK_PROMPT | self.llm.with_structured_output(RelevanceCheck)
        self.support_chain = SUPPORT_CHECK_PROMPT | self.llm.with_structured_output(SupportCheck)

    def should_retrieve(self, query: str) -> RetrievalDecision:
        """
        Decide if retrieval is needed for this query.

        [Retrieve] reflection token equivalent.
        """
        try:
            decision = self.retrieval_chain.invoke({"query": query})
            logger.info(f"Self-RAG Retrieve Decision: {decision.should_retrieve} ({decision.query_type})")
            return decision
        except Exception as e:
            logger.warning(f"Retrieval decision failed: {e}, defaulting to retrieve")
            return RetrievalDecision(
                should_retrieve=True, reason=f"Default to retrieve due to error: {e}", query_type="factual"
            )

    def check_relevance(self, query: str, documents: str) -> RelevanceCheck:
        """
        Check if retrieved documents are relevant.

        [IsRel] reflection token equivalent.
        """
        try:
            check = self.relevance_chain.invoke(
                {
                    "query": query,
                    "documents": documents[:3000],  # Limit length
                }
            )
            logger.info(f"Self-RAG Relevance: {check.is_relevant} (score: {check.relevance_score:.2f})")
            return check
        except Exception as e:
            logger.warning(f"Relevance check failed: {e}")
            return RelevanceCheck(is_relevant=True, relevance_score=0.5, critique=f"Check failed: {e}")

    def check_support(self, query: str, documents: str, answer: str) -> SupportCheck:
        """
        Check if generated answer is supported by documents.

        [IsSup] reflection token equivalent.
        """
        try:
            check = self.support_chain.invoke({"query": query, "documents": documents[:3000], "answer": answer[:1000]})
            logger.info(f"Self-RAG Support: {check.support_level}")
            return check
        except Exception as e:
            logger.warning(f"Support check failed: {e}")
            return SupportCheck(is_supported=True, support_level="partial", unsupported_claims=f"Check failed: {e}")

    def adaptive_generate(self, query: str, retrieve_fn, generate_fn) -> tuple[str, dict[str, Any]]:
        """
        Full Self-RAG pipeline with adaptive retrieval and verification.

        Args:
            query: User query
            retrieve_fn: Function that takes query and returns context string
            generate_fn: Function that takes (query, context) and returns answer

        Returns:
            Tuple of (answer, metadata)
        """
        metadata = {"self_rag": {}}

        # Step 1: Decide if retrieval is needed
        decision = self.should_retrieve(query)
        metadata["self_rag"]["retrieval_decision"] = decision.model_dump()

        context = ""
        if decision.should_retrieve:
            # Step 2: Retrieve
            context = retrieve_fn(query)

            # Step 3: Check relevance
            relevance = self.check_relevance(query, context)
            metadata["self_rag"]["relevance"] = relevance.model_dump()

            # If not relevant, try web search or proceed without context
            if not relevance.is_relevant or relevance.relevance_score < 0.3:
                logger.warning("Retrieved docs not relevant, proceeding with limited context")
                # Could trigger web search here as fallback

        # Step 4: Generate answer
        answer = generate_fn(query, context)

        # Step 5: Check support (only if we retrieved)
        if context:
            support = self.check_support(query, context, answer)
            metadata["self_rag"]["support"] = support.model_dump()

            # If not supported, could regenerate or flag
            if support.support_level == "none":
                logger.warning(f"Answer not supported: {support.unsupported_claims}")

        return answer, metadata


# Global instance
_self_rag: SelfRAG = None


def get_self_rag() -> SelfRAG:
    global _self_rag
    if _self_rag is None:
        _self_rag = SelfRAG()
    return _self_rag
