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
    ("system", """You are a helpful Pokemon assistant. Generate a draft answer to the question.
Include your confidence level (0-1) and brief reasoning.
Be concise but accurate. Use the provided context if available."""),
    ("human", """Context: {context}

Question: {query}

Generate a draft answer with confidence score and reasoning.""")
])


VERIFY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an expert verifier. Given multiple draft answers to a question,
select the best one and optionally refine it.

Criteria for selection:
1. Factual accuracy (most important)
2. Completeness
3. Clarity
4. Confidence alignment (high confidence should match quality)"""),
    ("human", """Question: {query}

Context: {context}

Draft Answers:
{drafts}

Select the best draft (by index starting from 0), provide the final answer (may refine), and explain your choice.""")
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
