"""
LightRAG Wrapper for Pokemon Knowledge Base

基于 lightrag-hku 实现的知识图谱 RAG 系统。
"""

import os
from typing import Any

from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc

from src.core.settings import settings
from src.utils.logger import LogManager

logger = LogManager()


class PokemonLightRAG:
    """
    LightRAG wrapper for Pokemon knowledge base.

    Provides async insert/query APIs with Neo4j/Milvus storage support.
    """

    def __init__(
        self,
        working_dir: str = None,
        workspace: str = "pokemon_kb",
        llm_config: dict[str, Any] = None,
        embedding_config: dict[str, Any] = None,
        storage_config: dict[str, Any] = None,
    ):
        """
        Initialize PokemonLightRAG.

        Args:
            working_dir: Directory for LightRAG data files
            workspace: Workspace/database identifier
            llm_config: LLM configuration (api_key, base_url, model)
            embedding_config: Embedding configuration
            storage_config: Storage backend config (neo4j, milvus)
        """
        self.working_dir = working_dir or os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "resources", "lightrag_data"
        )
        self.workspace = workspace

        # Default configs from settings
        self.llm_config = llm_config or {
            "api_key": settings.llm.api_key,
            "base_url": settings.llm.api_base,
            "model": settings.llm.model_name,
        }

        self.embedding_config = embedding_config or {
            "api_key": settings.embedding.api_key,
            "base_url": settings.embedding.api_base,
            "model": settings.embedding.model_name,
            "dimension": settings.embedding.dimension,
        }

        self.storage_config = storage_config or {
            "graph_storage": "Neo4JStorage",
            "vector_storage": "MilvusVectorDBStorage",
            "kv_storage": "JsonKVStorage",
            "doc_status_storage": "JsonDocStatusStorage",
        }

        self._rag: LightRAG | None = None
        self._initialized = False

        # Ensure working directory exists
        os.makedirs(self.working_dir, exist_ok=True)

        logger.info(f"PokemonLightRAG initialized with workspace: {workspace}")

    async def initialize(self) -> None:
        """Initialize LightRAG instance and storage backends."""
        if self._initialized:
            return

        try:
            self._rag = LightRAG(
                working_dir=self.working_dir,
                workspace=self.workspace,
                llm_model_func=self._get_llm_func(),
                embedding_func=self._get_embedding_func(),
                graph_storage=self.storage_config.get("graph_storage", "Neo4JStorage"),
                vector_storage=self.storage_config.get("vector_storage", "MilvusVectorDBStorage"),
                kv_storage=self.storage_config.get("kv_storage", "JsonKVStorage"),
                doc_status_storage=self.storage_config.get("doc_status_storage", "JsonDocStatusStorage"),
                log_file_path=os.path.join(self.working_dir, "lightrag.log"),
            )

            await self._rag.initialize_storages()
            self._initialized = True
            logger.info("LightRAG storage initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize LightRAG: {e}")
            raise

    def _get_llm_func(self):
        """Get LLM function for LightRAG."""
        model = self.llm_config.get("model")
        api_key = self.llm_config.get("api_key")
        base_url = self.llm_config.get("base_url")

        async def llm_model_func(
            prompt: str, system_prompt: str = None, history_messages: list = None, **kwargs
        ) -> str:
            return await openai_complete_if_cache(
                model=model,
                prompt=prompt,
                system_prompt=system_prompt,
                history_messages=history_messages or [],
                api_key=api_key,
                base_url=base_url,
                **kwargs,
            )

        return llm_model_func

    def _get_embedding_func(self) -> EmbeddingFunc:
        """Get embedding function for LightRAG."""
        model = self.embedding_config.get("model")
        api_key = self.embedding_config.get("api_key")
        base_url = self.embedding_config.get("base_url")
        dimension = self.embedding_config.get("dimension", 1024)

        return EmbeddingFunc(
            embedding_dim=dimension,
            max_token_size=8192,
            func=lambda texts: openai_embed(
                texts=texts,
                model=model,
                api_key=api_key,
                base_url=base_url.replace("/embeddings", ""),
            ),
        )

    async def insert(self, text: str, file_id: str = None, file_path: str = None, **kwargs) -> None:
        """
        Insert text into the knowledge base.

        Args:
            text: Text content to insert
            file_id: Optional file identifier
            file_path: Optional file path for reference
        """
        if not self._initialized:
            await self.initialize()

        try:
            await self._rag.ainsert(input=text, ids=file_id, file_paths=file_path, **kwargs)
            logger.info(f"Inserted content into LightRAG (file_id={file_id})")
        except Exception as e:
            logger.error(f"Failed to insert into LightRAG: {e}")
            raise

    async def query(
        self,
        query_text: str,
        mode: str = "mix",
        only_need_context: bool = False,
        top_k: int = 10,
        stream: bool = False,
        **kwargs,
    ) -> Any:
        """
        Query the knowledge base.

        Args:
            query_text: Query string
            mode: Query mode - naive, local, global, hybrid, mix
            only_need_context: If True, return context only without LLM answer
            top_k: Number of results to return
            stream: If True, return streaming response

        Returns:
            Query response (string or dict depending on mode)
        """
        if not self._initialized:
            await self.initialize()

        try:
            param = QueryParam(mode=mode, only_need_context=only_need_context, top_k=top_k, stream=stream, **kwargs)

            response = await self._rag.aquery(query_text, param)
            logger.debug(f"Query response: {str(response)[:200]}...")
            return response

        except Exception as e:
            logger.error(f"Query failed: {e}")
            return ""

    async def query_with_data(self, query_text: str, mode: str = "mix", top_k: int = 10, **kwargs) -> dict[str, Any]:
        """
        Query and return structured data (entities, relationships, chunks).

        Args:
            query_text: Query string
            mode: Query mode
            top_k: Number of results

        Returns:
            Dict with 'entities', 'relationships', 'chunks' keys
        """
        if not self._initialized:
            await self.initialize()

        try:
            param = QueryParam(mode=mode, only_need_context=True, top_k=top_k, **kwargs)

            response = await self._rag.aquery_data(query_text, param)
            return response.get("data", {})

        except Exception as e:
            logger.error(f"Query with data failed: {e}")
            return {"entities": [], "relationships": [], "chunks": []}

    async def delete_by_doc_id(self, doc_id: str) -> None:
        """
        Delete document by ID.

        Args:
            doc_id: Document identifier
        """
        if not self._initialized:
            await self.initialize()

        try:
            await self._rag.adelete_by_doc_id(doc_id)
            logger.info(f"Deleted document {doc_id} from LightRAG")
        except Exception as e:
            logger.error(f"Failed to delete document: {e}")
            raise

    async def finalize(self) -> None:
        """Finalize and cleanup storage connections."""
        if self._rag and self._initialized:
            try:
                await self._rag.finalize_storages()
                self._initialized = False
                logger.info("LightRAG storage finalized")
            except Exception as e:
                logger.error(f"Failed to finalize storage: {e}")

    @property
    def rag(self) -> LightRAG | None:
        """Access underlying LightRAG instance."""
        return self._rag


# Singleton instance for convenience
_default_instance: PokemonLightRAG | None = None


def get_lightrag_instance(workspace: str = "pokemon_kb", **kwargs) -> PokemonLightRAG:
    """
    Get or create a PokemonLightRAG instance.

    Args:
        workspace: Workspace identifier
        **kwargs: Additional configuration

    Returns:
        PokemonLightRAG instance
    """
    global _default_instance

    if _default_instance is None or _default_instance.workspace != workspace:
        _default_instance = PokemonLightRAG(workspace=workspace, **kwargs)

    return _default_instance
