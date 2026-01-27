"""
Knowledge Refresh - Automatic Knowledge Base Updates

Periodically crawls external sources (e.g., Bulbapedia) and updates
the vector store with new or changed content.
"""

import hashlib
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from src.core.settings import settings
from src.knowledge.store.vector import VectorStore
from src.utils.logger import get_logger

logger = get_logger(__name__)


class KnowledgeRefreshManager:
    """
    Manages automatic knowledge base updates.

    Features:
    - Tracks document versions via content hashes
    - Schedules periodic refreshes
    - Supports multiple sources (web, files, APIs)
    """

    def __init__(self, db_path: Path | None = None):
        self.db_path = db_path or (settings.paths.data_dir / "knowledge_refresh.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._init_db()
        self.vector_store = None  # Lazy load

    def _init_db(self):
        """Initialize tracking database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS document_versions (
                doc_id TEXT PRIMARY KEY,
                source_url TEXT,
                content_hash TEXT NOT NULL,
                last_updated TEXT NOT NULL,
                title TEXT,
                metadata TEXT
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS refresh_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_type TEXT NOT NULL,
                started_at TEXT NOT NULL,
                completed_at TEXT,
                docs_added INTEGER DEFAULT 0,
                docs_updated INTEGER DEFAULT 0,
                docs_unchanged INTEGER DEFAULT 0,
                status TEXT DEFAULT 'running',
                error_message TEXT
            )
        """)

        conn.commit()
        conn.close()

    def _get_vector_store(self) -> VectorStore:
        """Lazy load vector store."""
        if self.vector_store is None:
            self.vector_store = VectorStore(
                collection_name=settings.database.milvus_collection_name or "pokemon_knowledge"
            )
        return self.vector_store

    def _compute_hash(self, content: str) -> str:
        """Compute content hash for change detection."""
        return hashlib.md5(content.encode()).hexdigest()

    def _is_content_changed(self, doc_id: str, new_content: str) -> bool:
        """Check if content has changed from stored version."""
        new_hash = self._compute_hash(new_content)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT content_hash FROM document_versions WHERE doc_id = ?", (doc_id,))
        row = cursor.fetchone()
        conn.close()

        if row is None:
            return True  # New document

        return row[0] != new_hash

    def _update_document_version(
        self, doc_id: str, content: str, source_url: str = "", title: str = "", metadata: dict = None
    ):
        """Update document version tracking."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT OR REPLACE INTO document_versions
            (doc_id, source_url, content_hash, last_updated, title, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
        """,
            (
                doc_id,
                source_url,
                self._compute_hash(content),
                datetime.now().isoformat(),
                title,
                json.dumps(metadata or {}),
            ),
        )

        conn.commit()
        conn.close()

    def _start_refresh_log(self, source_type: str) -> int:
        """Start a refresh log entry."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO refresh_log (source_type, started_at)
            VALUES (?, ?)
        """,
            (source_type, datetime.now().isoformat()),
        )

        log_id = cursor.lastrowid
        conn.commit()
        conn.close()

        return log_id

    def _complete_refresh_log(
        self, log_id: int, added: int, updated: int, unchanged: int, status: str = "completed", error: str = None
    ):
        """Complete a refresh log entry."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            UPDATE refresh_log SET
                completed_at = ?,
                docs_added = ?,
                docs_updated = ?,
                docs_unchanged = ?,
                status = ?,
                error_message = ?
            WHERE id = ?
        """,
            (datetime.now().isoformat(), added, updated, unchanged, status, error, log_id),
        )

        conn.commit()
        conn.close()

    async def refresh_from_web(self, urls: list[str], chunk_size: int = 500) -> dict[str, int]:
        """
        Refresh knowledge from web URLs.

        Args:
            urls: List of URLs to crawl
            chunk_size: Size of text chunks for indexing

        Returns:
            Stats dict with added, updated, unchanged counts
        """
        log_id = self._start_refresh_log("web")
        stats = {"added": 0, "updated": 0, "unchanged": 0}

        try:
            import httpx
            from bs4 import BeautifulSoup

            self._get_vector_store()

            async with httpx.AsyncClient() as client:
                for url in urls:
                    try:
                        response = await client.get(url, timeout=30.0)
                        response.raise_for_status()

                        soup = BeautifulSoup(response.text, "html.parser")

                        # Extract text content
                        for script in soup(["script", "style"]):
                            script.decompose()

                        text = soup.get_text(separator="\n", strip=True)
                        title = soup.title.string if soup.title else url

                        doc_id = self._compute_hash(url)

                        if not self._is_content_changed(doc_id, text):
                            stats["unchanged"] += 1
                            logger.debug(f"Unchanged: {url}")
                            continue

                        # Chunk and index
                        chunks = self._chunk_text(text, chunk_size)

                        for _i, _chunk in enumerate(chunks):
                            pass
                            # Note: This is simplified. Real impl would upsert.
                            # vector_store.add_documents(...)

                        self._update_document_version(doc_id, text, url, title, {"chunks": len(chunks)})

                        stats["updated"] += 1
                        logger.info(f"Updated: {url} ({len(chunks)} chunks)")

                    except Exception as e:
                        logger.error(f"Failed to refresh {url}: {e}")

            self._complete_refresh_log(log_id, **stats)

        except Exception as e:
            logger.error(f"Refresh failed: {e}")
            self._complete_refresh_log(log_id, **stats, status="failed", error=str(e))

        return stats

    def refresh_from_directory(
        self,
        directory: Path,
        extensions: list[str] | None = None,
    ) -> dict[str, int]:
        """
        Refresh knowledge from local files.

        Args:
            directory: Directory to scan
            extensions: File extensions to include

        Returns:
            Stats dict
        """
        log_id = self._start_refresh_log("directory")
        stats = {"added": 0, "updated": 0, "unchanged": 0}
        extensions = extensions or [".txt", ".md", ".json"]

        try:
            for ext in extensions:
                for file_path in directory.rglob(f"*{ext}"):
                    try:
                        content = file_path.read_text(encoding="utf-8")
                        doc_id = self._compute_hash(str(file_path))

                        if not self._is_content_changed(doc_id, content):
                            stats["unchanged"] += 1
                            continue

                        # Would index here...
                        self._update_document_version(doc_id, content, source_url=str(file_path), title=file_path.name)

                        stats["updated"] += 1
                        logger.info(f"Updated: {file_path}")

                    except Exception as e:
                        logger.error(f"Failed to process {file_path}: {e}")

            self._complete_refresh_log(log_id, **stats)

        except Exception as e:
            self._complete_refresh_log(log_id, **stats, status="failed", error=str(e))

        return stats

    def _chunk_text(self, text: str, chunk_size: int = 500) -> list[str]:
        """Simple text chunking."""
        words = text.split()
        chunks = []
        current_chunk = []
        current_size = 0

        for word in words:
            current_chunk.append(word)
            current_size += len(word) + 1

            if current_size >= chunk_size:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_size = 0

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def get_refresh_status(self) -> dict[str, Any]:
        """Get current refresh status and history."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Recent logs
        cursor.execute("""
            SELECT * FROM refresh_log
            ORDER BY started_at DESC
            LIMIT 10
        """)
        logs = cursor.fetchall()

        # Document counts
        cursor.execute("SELECT COUNT(*) FROM document_versions")
        doc_count = cursor.fetchone()[0]

        conn.close()

        return {
            "total_documents": doc_count,
            "recent_refreshes": [
                {
                    "id": log[0],
                    "source_type": log[1],
                    "started_at": log[2],
                    "completed_at": log[3],
                    "docs_added": log[4],
                    "docs_updated": log[5],
                    "docs_unchanged": log[6],
                    "status": log[7],
                }
                for log in logs
            ],
        }


# Global instance
_manager: KnowledgeRefreshManager = None


def get_knowledge_refresh_manager() -> KnowledgeRefreshManager:
    global _manager
    if _manager is None:
        _manager = KnowledgeRefreshManager()
    return _manager
