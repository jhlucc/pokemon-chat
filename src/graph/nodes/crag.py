"""
Corrective RAG (CRAG) Evaluator

Evaluates retrieval quality and determines if:
- CORRECT: Use retrieved docs as-is
- AMBIGUOUS: Supplement with web search
- WRONG: Discard and use only web search
"""

from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from src.core.llm_factory import build_chat_llm
from src.utils.logger import get_logger

logger = get_logger(__name__)


class RetrievalGrade(BaseModel):
    """Grade for retrieval quality."""

    grade: Literal["CORRECT", "AMBIGUOUS", "WRONG"] = Field(
        description="Grade of retrieval quality: CORRECT if docs answer the query, AMBIGUOUS if partially relevant, WRONG if irrelevant"
    )
    reason: str = Field(description="Brief explanation for the grade")


GRADING_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """你是宝可梦知识检索质量评估员。
给定用户问题和检索到的文档，评估文档是否相关且足以回答问题。

评分标准：
- CORRECT：文档直接回答了问题
  例：问"皮卡丘属性"，文档明确说"电系"
- AMBIGUOUS：文档部分相关，可能需要补充
  例：问"皮卡丘对抗岩系怎么样"，文档只说了皮卡丘是电系
- WRONG：文档与问题无关
  例：问"皮卡丘"，文档讲的是"喇叭芽"

宝可梦特殊考量：
- 如果文档讲的是进化链上的相关宝可梦，可以算AMBIGUOUS
- 如果文档讲的是相同属性的其他宝可梦，可以算AMBIGUOUS
- 如果文档完全距离，算WRONG""",
        ),
        (
            "human",
            """用户问题: {query}

检索到的宝可梦知识:
{documents}

请评估检索质量 (CORRECT/AMBIGUOUS/WRONG) 并简要说明理由。""",
        ),
    ]
)


class CRAGEvaluator:
    """
    Corrective RAG Evaluator.

    Grades retrieval quality and determines the correction strategy.
    """

    def __init__(self):
        # Use unified LLM factory for provider-agnostic LLM
        self.llm = build_chat_llm(temperature=0)
        self.grading_chain = GRADING_PROMPT | self.llm.with_structured_output(RetrievalGrade)

    def grade(self, query: str, documents: list[str]) -> RetrievalGrade:
        """
        Grade the quality of retrieved documents for a query.

        Args:
            query: User's original query
            documents: List of retrieved document contents

        Returns:
            RetrievalGrade with grade and reason
        """
        if not documents:
            return RetrievalGrade(grade="WRONG", reason="No documents retrieved")

        try:
            docs_text = "\n\n---\n\n".join([f"Doc {i + 1}: {doc[:500]}..." for i, doc in enumerate(documents[:5])])

            result = self.grading_chain.invoke({"query": query, "documents": docs_text})

            logger.info(f"CRAG Grade: {result.grade} - {result.reason}")
            return result

        except Exception as e:
            logger.error(f"CRAG grading failed: {e}")
            # Default to AMBIGUOUS on error (safe fallback)
            return RetrievalGrade(grade="AMBIGUOUS", reason=f"Grading error: {e}")

    def correct(self, grade: RetrievalGrade, original_docs: list[str], web_search_fn) -> list[str]:
        """
        Apply correction strategy based on grade.

        Args:
            grade: RetrievalGrade from grade()
            original_docs: Originally retrieved documents
            web_search_fn: Callable that performs web search and returns List[str]

        Returns:
            Corrected list of documents
        """
        if grade.grade == "CORRECT":
            return original_docs

        elif grade.grade == "AMBIGUOUS":
            # Supplement with web search
            try:
                web_docs = web_search_fn()
                return original_docs + web_docs
            except Exception as e:
                logger.warning(f"Web search failed during correction: {e}")
                return original_docs

        else:  # WRONG
            # Discard original, use only web search
            try:
                return web_search_fn()
            except Exception as e:
                logger.warning(f"Web search failed during correction: {e}")
                return original_docs  # Fallback to original


# Global instance
_evaluator: CRAGEvaluator = None


def get_crag_evaluator() -> CRAGEvaluator:
    global _evaluator
    if _evaluator is None:
        _evaluator = CRAGEvaluator()
    return _evaluator
