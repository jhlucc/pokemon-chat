"""
Query Decomposition for Complex Questions

Breaks down complex queries into simpler sub-queries,
retrieves for each, and merges results.
"""
from typing import List, Tuple
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from src.core.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SubQueries(BaseModel):
    """Decomposed sub-queries."""
    queries: List[str] = Field(
        description="List of simpler sub-queries that together answer the original question"
    )
    reasoning: str = Field(
        description="Brief explanation of why these sub-queries were chosen"
    )


DECOMPOSITION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a query decomposition expert. 
Given a complex question, break it down into 2-4 simpler sub-questions that can be answered independently.

Rules:
1. Each sub-query should be self-contained and answerable on its own
2. Sub-queries should cover different aspects of the original question
3. For simple questions, just return the original question as the only sub-query
4. Focus on factual sub-queries that can be resolved with knowledge retrieval

Examples:
- "Compare Pikachu and Charmander's stats" -> ["What are Pikachu's stats?", "What are Charmander's stats?"]
- "Who is stronger, Ash's Pikachu or Gary's Blastoise?" -> ["What are the abilities of Ash's Pikachu?", "What are the abilities of Gary's Blastoise?", "What is the type effectiveness between Electric and Water?"]
- "Tell me about Pikachu" -> ["Tell me about Pikachu"] (simple, no decomposition needed)"""),
    ("human", "Question: {query}\n\nDecompose this into sub-queries:")
])


class QueryDecomposer:
    """
    Decomposes complex queries into simpler sub-queries.
    """
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model=settings.llm.model_name,
            api_key=settings.llm.api_key,
            base_url=settings.llm.api_base,
            temperature=0
        )
        self.chain = DECOMPOSITION_PROMPT | self.llm.with_structured_output(SubQueries)
    
    def decompose(self, query: str) -> List[str]:
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
