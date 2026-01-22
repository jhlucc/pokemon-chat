import unittest
from unittest.mock import MagicMock, patch
from src.knowledge.store.vector import VectorStore
from src.tests.mock_hybrid_embedding import MockHybridEmbedding

class TestHybridSearch(unittest.TestCase):
    
    @patch('src.knowledge.store.vector.connections')
    @patch('src.knowledge.store.vector.Collection')
    @patch('src.knowledge.store.vector.utility')
    def test_hybrid_search_logic(self, MockUtility, MockCollection, MockConnections):
        # Mock Milvus connection check
        MockConnections.has_connection.return_value = True
        
        # Mock Collection existence
        MockUtility.has_collection.return_value = False
        
        # Init Store
        store = VectorStore(
            collection_name="test_hybrid",
            embedding_model=MockHybridEmbedding(),
            overwrite=True
        )
        
        # Mock hybrid_search return
        mock_hit = MagicMock()
        mock_hit.score = 0.9
        mock_hit.entity.get.side_effect = lambda k, d=None: "Test Content" if k == "text" else {}
        
        store.collection.hybrid_search.return_value = [[mock_hit]]
        
        # Execute
        results = store.hybrid_search(
            query="Pikachu",
            sparse_embedding={1: 0.5, 2: 0.3}
        )
        
        # Verify
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].page_content, "Test Content")
        store.collection.hybrid_search.assert_called()

if __name__ == "__main__":
    unittest.main()
