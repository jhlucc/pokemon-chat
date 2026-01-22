from typing import Any, Dict, List, Optional
from langchain_community.graphs import Neo4jGraph
from src.core.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

class GraphStore:
    """
    Unified Graph Store for Knowledge Graph (Neo4j).
    Wraps langchain_community.graphs.Neo4jGraph for easier integration.
    """
    def __init__(self, refresh_schema: bool = True):
        if not settings.features.enable_knowledge_graph:
            logger.warning("Knowledge Graph feature is disabled in settings.")
            self.graph = None
            return

        try:
            self.graph = Neo4jGraph(
                url=settings.database.neo4j_uri,
                username=settings.database.neo4j_username,
                password=settings.database.neo4j_password,
                refresh_schema=refresh_schema
            )
            logger.info("Connected to Neo4j GraphStore.")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            self.graph = None

    def query(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Execute a Cypher query."""
        if not self.graph:
            return []
        return self.graph.query(query, params)

    def refresh_schema(self):
        """Refreshes the Neo4j graph schema information."""
        if self.graph:
            self.graph.refresh_schema()

    def get_structured_schema(self) -> Dict[str, Any]:
        """Returns the structured schema of the graph."""
        if not self.graph:
            return {}
        return self.graph.get_structured_schema
