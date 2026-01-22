"""
Speculative RAG - Draft and Verify Pattern

Uses a smaller, faster model to generate multiple draft answers,
then a larger model verifies and selects the best one.
This reduces latency by avoiding multiple large model calls.
"""
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from src.core.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DraftAnswer(BaseModel):
    """A draft answer from the drafter model."""
    answer: str
    confidence: float = Field(ge=0, le=1, description="Self-assessed confidence")
    reasoning: str = Field(description="Brief reasoning for the answer")


class VerificationResult(BaseModel):
    """Verification result from the verifier model."""
    selected_index: int = Field(description="Index of the best draft (0-based)")
    final_answer: str = Field(description="The final, potentially refined answer")
    verification_notes: str = Field(description="Notes on why this was selected")


DRAFT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """你是宝可梦知识草案生成器。根据用户问题和参考资料，生成一个草案回答。

回答要求：
- 使用准确的宝可梦术语（属性、特性、技能、种族值等）
- 如果涉及数值，确保准确
- 简洁但全面
- 评估自己的信心程度（0-1）

信心评分标准：
- 0.9+: 文档直接支持回答
- 0.7-0.9: 文档部分支持，需要推理
- 0.5-0.7: 主要基于常识
- <0.5: 不确定"""),
    ("human", """参考资料: {context}

用户问题: {query}

请生成草案回答，并评估信心度和简要理由。""")
])


VERIFY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """你是宝可梦知识专家审核员。给定多个草案回答，选择最佳的一个并可选择性地优化它。

评估标准（按重要性排序）：
1. 宝可梦信息准确性（最重要）
   - 属性、类型、进化链是否正确
   - 数值数据是否精确
2. 回答完整性
   - 是否覆盖了问题的各个方面
3. 表达清晰度
   - 是否易于理解
4. 信心与质量匹配
   - 高信心的回答质量应该也高

如果多个草案质量相似，选择最全面的那个。"""),
    ("human", """用户问题: {query}

参考资料: {context}

候选草案:
{drafts}

请选择最佳草案（按索引0开始），给出最终回答（可以优化），并解释选择理由。""")
])


class SpeculativeRAG:
    """
    Speculative RAG implementation.
    
    Uses a fast drafter model to generate multiple candidates,
    then a stronger verifier model to select/refine the best answer.
    """
    
    def __init__(
        self,
        drafter_model: Optional[str] = None,
        verifier_model: Optional[str] = None,
        num_drafts: int = 3
    ):
        # Drafter: smaller, faster model
        self.drafter = ChatOpenAI(
            model=drafter_model or settings.llm.model_name_lite or settings.llm.model_name,
            api_key=settings.llm.api_key,
            base_url=settings.llm.api_base,
            temperature=0.7  # Higher temp for diversity
        )
        
        # Verifier: larger, more capable model
        self.verifier = ChatOpenAI(
            model=verifier_model or settings.llm.model_name,
            api_key=settings.llm.api_key,
            base_url=settings.llm.api_base,
            temperature=0
        )
        
        self.num_drafts = num_drafts
        
        self.draft_chain = DRAFT_PROMPT | self.drafter.with_structured_output(DraftAnswer)
        self.verify_chain = VERIFY_PROMPT | self.verifier.with_structured_output(VerificationResult)
    
    def generate_drafts(self, query: str, context: str) -> List[DraftAnswer]:
        """Generate multiple draft answers in parallel."""
        drafts = []
        
        # Use ThreadPoolExecutor for parallel generation
        with ThreadPoolExecutor(max_workers=self.num_drafts) as executor:
            futures = [
                executor.submit(
                    self.draft_chain.invoke,
                    {"query": query, "context": context}
                )
                for _ in range(self.num_drafts)
            ]
            
            for future in as_completed(futures):
                try:
                    draft = future.result()
                    drafts.append(draft)
                except Exception as e:
                    logger.warning(f"Draft generation failed: {e}")
        
        return drafts
    
    def verify_and_select(
        self,
        query: str,
        context: str,
        drafts: List[DraftAnswer]
    ) -> VerificationResult:
        """Verify drafts and select the best one."""
        if not drafts:
            return VerificationResult(
                selected_index=-1,
                final_answer="Unable to generate an answer.",
                verification_notes="No drafts available"
            )
        
        # Format drafts for verification
        drafts_text = "\n\n".join([
            f"[Draft {i}] (Confidence: {d.confidence:.2f})\n{d.answer}\nReasoning: {d.reasoning}"
            for i, d in enumerate(drafts)
        ])
        
        try:
            result = self.verify_chain.invoke({
                "query": query,
                "context": context,
                "drafts": drafts_text
            })
            return result
        except Exception as e:
            logger.error(f"Verification failed: {e}")
            # Fallback: return highest confidence draft
            best_draft = max(drafts, key=lambda d: d.confidence)
            return VerificationResult(
                selected_index=drafts.index(best_draft),
                final_answer=best_draft.answer,
                verification_notes=f"Fallback: selected highest confidence draft due to error: {e}"
            )
    
    def generate(self, query: str, context: str) -> Tuple[str, Dict[str, Any]]:
        """
        Main generation method using speculative decoding pattern.
        
        Returns:
            Tuple of (final_answer, metadata)
        """
        logger.info(f"Speculative RAG: Generating {self.num_drafts} drafts")
        
        # Step 1: Generate drafts (parallel, fast)
        drafts = self.generate_drafts(query, context)
        logger.info(f"Generated {len(drafts)} drafts")
        
        # Step 2: Verify and select (single call to strong model)
        result = self.verify_and_select(query, context, drafts)
        
        metadata = {
            "num_drafts": len(drafts),
            "selected_index": result.selected_index,
            "verification_notes": result.verification_notes,
            "draft_confidences": [d.confidence for d in drafts]
        }
        
        logger.info(f"Selected draft {result.selected_index}, final answer length: {len(result.final_answer)}")
        
        return result.final_answer, metadata


# Global instance
_speculative_rag: SpeculativeRAG = None

def get_speculative_rag() -> SpeculativeRAG:
    global _speculative_rag
    if _speculative_rag is None:
        _speculative_rag = SpeculativeRAG()
    return _speculative_rag
