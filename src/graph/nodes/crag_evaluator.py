"""
Corrective RAG (CRAG) Evaluator

Evaluates retrieval quality and determines if:
- CORRECT: Use retrieved docs as-is
- AMBIGUOUS: Supplement with web search
- WRONG: Discard and use only web search
"""
from typing import Dict, Any, List, Literal
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field

from src.core.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

class RetrievalGrade(BaseModel):
    """Grade for retrieval quality."""
    grade: Literal["CORRECT", "AMBIGUOUS", "WRONG"] = Field(
        description="Grade of retrieval quality: CORRECT if docs answer the query, AMBIGUOUS if partially relevant, WRONG if irrelevant"
    )
    reason: str = Field(description="Brief explanation for the grade")


GRADING_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a retrieval quality evaluator. 
Given a user query and retrieved documents, evaluate if the documents are relevant and sufficient to answer the query.

Grade as:
- CORRECT: Documents directly answer the query with high confidence
- AMBIGUOUS: Documents are partially relevant but may need supplementation
- WRONG: Documents are irrelevant or do not help answer the query

Be strict but fair. If the query is about a specific topic and documents mention it, that's usually CORRECT.
If documents are about related but different topics, that's AMBIGUOUS.
If documents are completely off-topic, that's WRONG."""),
    ("human", """Query: {query}

Retrieved Documents:
{documents}

Provide your grade (CORRECT/AMBIGUOUS/WRONG) and a brief reason.""")
])


class CRAGEvaluator:
    """
    Corrective RAG Evaluator.
    
    Grades retrieval quality and determines the correction strategy.
    """
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model=settings.llm.model_name,
            api_key=settings.llm.api_key,
            base_url=settings.llm.api_base,
            temperature=0
        )
        self.grading_chain = GRADING_PROMPT | self.llm.with_structured_output(RetrievalGrade)
    
    def grade(self, query: str, documents: List[str]) -> RetrievalGrade:
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
            docs_text = "\n\n---\n\n".join([f"Doc {i+1}: {doc[:500]}..." for i, doc in enumerate(documents[:5])])
            
            result = self.grading_chain.invoke({
                "query": query,
                "documents": docs_text
            })
            
            logger.info(f"CRAG Grade: {result.grade} - {result.reason}")
            return result
            
        except Exception as e:
            logger.error(f"CRAG grading failed: {e}")
            # Default to AMBIGUOUS on error (safe fallback)
            return RetrievalGrade(grade="AMBIGUOUS", reason=f"Grading error: {e}")
    
    def correct(
        self,
        grade: RetrievalGrade,
        original_docs: List[str],
        web_search_fn
    ) -> List[str]:
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
