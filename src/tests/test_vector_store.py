import unittest
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from src.knowledge.store.vector import VectorStore


class TestVectorStore(unittest.TestCase):
    @patch("src.knowledge.store.vector.connections")
    @patch("src.knowledge.store.vector.Collection")
    @patch("src.knowledge.store.vector.utility")
    def test_initialization(self, mock_utility, mock_collection, mock_connections):
        mock_utility.has_collection.return_value = False
        mock_connections.has_connection.return_value = False

        VectorStore(collection_name="test_collection", embedding_model=MagicMock(), reranker_model=MagicMock())

        mock_connections.connect.assert_called()
        mock_utility.has_collection.assert_called_with("test_collection", using="default")
        mock_collection.assert_called()  # Creation called

    @patch("src.knowledge.store.vector.connections")
    @patch("src.knowledge.store.vector.Collection")
    @patch("src.knowledge.store.vector.utility")
    def test_insert_with_precomputed_embedding(self, mock_utility, mock_collection, mock_connections):
        mock_utility.has_collection.return_value = True

        store = VectorStore(collection_name="test")
        col_instance = mock_collection.return_value

        # Doc with embedding
        doc = Document(page_content="hello", metadata={"embedding": [0.1] * 1024})

        store.insert([doc])

        col_instance.insert.assert_called_once()
        args = col_instance.insert.call_args[0][0]
        # Check embeddings column (index 1)
        self.assertEqual(args[1][0], [0.1] * 1024)

    @patch("src.knowledge.store.vector.connections")
    @patch("src.knowledge.store.vector.Collection")
    @patch("src.knowledge.store.vector.utility")
    def test_insert_needs_embedding(self, mock_utility, mock_collection, mock_connections):
        mock_utility.has_collection.return_value = True

        mock_embedder = MagicMock()
        mock_embedder.embed_documents.return_value = [[0.2] * 1024]

        store = VectorStore(collection_name="test", embedding_model=mock_embedder)
        col_instance = mock_collection.return_value

        # Doc WITHOUT embedding
        doc = Document(page_content="hello")

        store.insert([doc])

        mock_embedder.embed_documents.assert_called_with(["hello"])
        col_instance.insert.assert_called_once()
        args = col_instance.insert.call_args[0][0]
        self.assertEqual(args[1][0], [0.2] * 1024)

    @patch("src.knowledge.store.vector.settings")
    @patch("src.knowledge.store.vector.connections")
    @patch("src.knowledge.store.vector.Collection")
    @patch("src.knowledge.store.vector.utility")
    def test_uses_settings_milvus_uri_when_default_host_port(
        self, mock_utility, mock_collection, mock_connections, mock_settings
    ):
        mock_utility.has_collection.return_value = False
        mock_connections.has_connection.return_value = False
        mock_settings.database.milvus_uri = "http://milvus:19530"

        VectorStore(collection_name="test_collection", embedding_model=MagicMock(), reranker_model=MagicMock())

        mock_connections.connect.assert_called_with(alias="default", host="milvus", port="19530")

    def test_rerank_sorts_by_rerank_score(self):
        store = VectorStore.__new__(VectorStore)  # bypass __init__ (no Milvus)

        class _DummyReranker:
            def run(self, _query, _docs, normalize=True):  # noqa: ANN001
                assert normalize is True
                # Higher is better: doc[1] should become first.
                return [0.2, 0.9, 0.1]

        store.reranker = _DummyReranker()

        docs = [
            Document(page_content="a", metadata={"score": 0.3}),
            Document(page_content="b", metadata={"score": 0.2}),
            Document(page_content="c", metadata={"score": 0.1}),
        ]

        out = store._rerank("q", docs, top_k=2)
        self.assertEqual([d.page_content for d in out], ["b", "a"])
        self.assertIn("rerank_score", out[0].metadata)


if __name__ == "__main__":
    unittest.main()
