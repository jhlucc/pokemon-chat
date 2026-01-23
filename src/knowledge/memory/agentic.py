"""
AgenticMemory - Using Mem0 (mem0ai)
"""
import logging
from typing import Optional, List
from mem0 import Memory
import os

from src.core.settings import settings

logger = logging.getLogger(__name__)

class AgenticMemory:
    """
    Wrapper around Mem0 for long-term user memory.
    """
    
    def __init__(self):
        # Configure Mem0
        # Ensure OpenAI API key is set for Mem0
        os.environ["OPENAI_API_KEY"] = settings.openai_api_key or ""
        if settings.openai_api_base:
            os.environ["OPENAI_BASE_URL"] = settings.openai_api_base
            
        self.client = Memory()

    def add_conversation_turn(self, user_id: str, role: str, content: str):
        """
        Add a conversation turn. 
        Mem0 extracts facts automatically.
        """
        try:
            # We can treat every turn as a potential memory source
            # Metadata can track role
            self.client.add(content, user_id=user_id, metadata={"role": role})
        except Exception as e:
            logger.error(f"Mem0 add failed: {e}")

    def extract_and_update_preferences(self, user_id: str):
        """
        No-op for Mem0 as it extracts on 'add'.
        Kept for compatibility with chat_router.
        """
        pass

    def get_system_prompt_injection(self, user_id: str) -> str:
        """
        Retrieve relevant memories and format as system prompt.
        """
        try:
            # Fetch all memories or perform a search based on context?
            # For system prompt, we usually want "core" facts.
            # get_all returns a list of dictionaries.
            memories = self.client.get_all(user_id=user_id, limit=20) 
            
            if not memories:
                return ""
            
            # Format memories
            # Mem0 result structure: [{'id':..., 'memory': 'User likes Python', ...}]
            facts = [m.get("memory", "") for m in memories]
            facts_str = "\n".join(f"- {f}" for f in facts if f)
            
            if not facts_str:
                return ""
                
            return f"\n\n[User Long-term Memory]\n{facts_str}"
            
        except Exception as e:
            logger.error(f"Mem0 retrieval failed: {e}")
            return ""

# Global instance
_memory: AgenticMemory = None

def get_agentic_memory() -> AgenticMemory:
    global _memory
    if _memory is None:
        _memory = AgenticMemory()
    return _memory

