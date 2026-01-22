import os
from typing import List, Dict, Any, Optional
try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma

from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

from src.core.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

class SemanticMemoryManager:
    def __init__(self, collection_name: str = "pokemon_chat_memory"):
        self.persist_directory = os.path.join(settings.paths.artifacts_data, "chroma_db")
        self.collection_name = collection_name
        
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small", 
            api_key=settings.llm.api_key,
            base_url=settings.llm.api_base
        )
        
        self.vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory
        )
        logger.info(f"SemanticMemoryManager initialized at {self.persist_directory}")

    async def add_memory(self, text: str, metadata: Dict[str, Any] = None):
        """Add a memory nugget to the store (async)"""
        try:
            doc = Document(page_content=text, metadata=metadata or {})
            await self.vector_store.aadd_documents([doc])
            logger.debug(f"Added memory: {text[:50]}...")
        except Exception as e:
            logger.error(f"Failed to add memory: {e}")

    def add_memory_sync(self, text: str, metadata: Dict[str, Any] = None):
        """Add a memory nugget to the store (sync)"""
        try:
            doc = Document(page_content=text, metadata=metadata or {})
            self.vector_store.add_documents([doc])
            logger.debug(f"Added memory: {text[:50]}...")
        except Exception as e:
            logger.error(f"Failed to add memory: {e}")

    async def search_memory(self, query: str, k: int = 3, filter: Optional[Dict] = None) -> List[Document]:
        """Search for relevant memories (async)"""
        try:
            results = await self.vector_store.asimilarity_search(query, k=k, filter=filter)
            return results
        except Exception as e:
            logger.error(f"Failed to search memory: {e}")
            return []

    def search_memory_sync(self, query: str, k: int = 3, filter: Optional[Dict] = None) -> List[Document]:
        """Search for relevant memories (sync)"""
        try:
            results = self.vector_store.similarity_search(query, k=k, filter=filter)
            return results
        except Exception as e:
            logger.error(f"Failed to search memory: {e}")
            return []

    async def clear_memory(self):
        """Clear all memories (use with caution)"""
        try:
            self.vector_store.delete_collection()
            logger.warning("Memory collection deleted")
            # Re-init
            self.vector_store = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory
            )
        except Exception as e:
            logger.error(f"Failed to clear memory: {e}")
