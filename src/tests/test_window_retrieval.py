import unittest
from unittest.mock import MagicMock, patch

from src.knowledge.store.vector import VectorStore


class TestWindowRetrieval(unittest.TestCase):
    @patch("src.knowledge.store.vector.connections")
    @patch("src.knowledge.store.vector.Collection")
    @patch("src.knowledge.store.vector.utility")
    def test_get_adjacent_chunks(self, MockUtility, MockCollection, MockConnections):
        MockConnections.has_connection.return_value = True
        store = VectorStore(embedding_model=MagicMock())

        # Mock query return
        store.collection.query.return_value = [
            {"text": "Chunk 4", "metadata": {"file_id": "f1", "chunk_index": 4}},
            {"text": "Chunk 5", "metadata": {"file_id": "f1", "chunk_index": 5}},
            {"text": "Chunk 6", "metadata": {"file_id": "f1", "chunk_index": 6}},
        ]

        docs = store.get_adjacent_chunks("f1", 5, radius=1)

        self.assertEqual(len(docs), 3)
        self.assertEqual(docs[0].page_content, "Chunk 4")
        store.collection.query.assert_called()


if __name__ == "__main__":
    unittest.main()
