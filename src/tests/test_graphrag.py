import unittest
from unittest.mock import MagicMock, patch

from langchain_core.messages import HumanMessage

from src.graph.nodes.graph_worker import GraphWorker
from src.graph.nodes.rag_worker import RagWorker


class TestGraphRAG(unittest.TestCase):
    @patch("src.graph.nodes.graph_worker.GraphStore")
    @patch("src.graph.nodes.graph_worker.GraphCypherQAChain")
    def test_graph_worker(self, MockChain, MockStore):
        # Mock Store and Graph
        mock_graph = MagicMock()
        MockStore.return_value.graph = mock_graph

        # Mock Chain
        mock_chain_instance = MockChain.from_llm.return_value
        mock_chain_instance.invoke.return_value = {"result": "Pikachu evolves into Raichu."}

        worker = GraphWorker()
        state = {"messages": [HumanMessage(content="What does Pikachu evolve into?")]}

        result = worker(state)

        self.assertEqual(result["messages"][0].content, "Pikachu evolves into Raichu.")
        MockChain.from_llm.assert_called()

    @patch("src.graph.nodes.rag_worker.VectorStore")
    @patch("src.graph.nodes.rag_worker.HyDEOperator")
    def test_rag_worker_hyde(self, MockHyDE, MockVectorStore):
        # Mock VectorStore
        mock_store = MockVectorStore.return_value
        mock_store.search.return_value = [MagicMock(page_content="Context")]

        # Mock HyDE
        mock_hyde_instance = MockHyDE.return_value
        mock_hyde_instance.call.return_value = "Hypothetical Answer"

        worker = RagWorker()
        # Mock LLM to avoid calls
        worker.llm = MagicMock()
        worker.llm.invoke.return_value.content = "Final Answer"

        state = {"messages": [HumanMessage(content="Test Query")]}

        worker(state)

        # Verify HyDE was called
        mock_hyde_instance.call.assert_called()
        # Verify Search called with HyDE result (Hypothetical Answer)
        mock_store.search.assert_called_with(
            "Hypothetical Answer", top_k=5, rerank=True
        )  # rerank defaults to settings (True)


if __name__ == "__main__":
    unittest.main()
