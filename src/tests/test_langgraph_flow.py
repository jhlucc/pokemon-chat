import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage

# We need to mock settings or env vars if they are accessed at import time?
# src.graph.workflow imports settings? No, but nodes do.
# Code is already imported.


class TestLangGraphFlow(unittest.TestCase):
    @patch("src.graph.nodes.supervisor.SupervisorNode")
    @patch("src.graph.nodes.rag_worker.RagWorker")
    @patch("src.graph.nodes.web_worker.WebWorker")
    @patch("src.graph.nodes.graph_worker.GraphWorker")
    @patch("src.graph.nodes.stats_worker.StatsWorker")
    def test_routing_rag(self, MockStats, MockGraph, MockWeb, MockRag, MockSupervisor):
        # Setup instances
        mock_sup_instance = MockSupervisor.return_value
        mock_rag_instance = MockRag.return_value

        # Supervisor routing: Rag -> Finish
        # Note: SupervisorNode return value is a dict {"next": ...}
        mock_sup_instance.side_effect = [{"next": "rag_worker"}, {"next": "FINISH"}]

        # Rag worker response
        mock_rag_instance.return_value = {"messages": [AIMessage(content="RAG Result")]}

        # Import graph here to ensure it uses the patches if meaningful?
        # Actually workflow has "from ... import supervisor_node".
        # supervisor_node calls SupervisorNode().
        # So patching the class SupervisorNode works.
        from src.graph.workflow import graph

        inputs = {"messages": [HumanMessage(content="Query")]}
        graph.invoke(inputs)

        self.assertEqual(mock_sup_instance.call_count, 2)
        mock_rag_instance.assert_called_once()
        MockWeb.return_value.assert_not_called()

    @patch("src.graph.nodes.supervisor.SupervisorNode")
    @patch("src.graph.nodes.rag_worker.RagWorker")
    @patch("src.graph.nodes.web_worker.WebWorker")
    @patch("src.graph.nodes.graph_worker.GraphWorker")
    @patch("src.graph.nodes.stats_worker.StatsWorker")
    def test_routing_web(self, MockStats, MockGraph, MockWeb, MockRag, MockSupervisor):
        mock_sup_instance = MockSupervisor.return_value
        mock_web_instance = MockWeb.return_value

        # Supervisor routing: Web -> Finish
        mock_sup_instance.side_effect = [{"next": "web_worker"}, {"next": "FINISH"}]

        mock_web_instance.return_value = {"messages": [AIMessage(content="Web Result")]}

        from src.graph.workflow import graph

        inputs = {"messages": [HumanMessage(content="Web Query")]}
        graph.invoke(inputs)

        self.assertEqual(mock_sup_instance.call_count, 2)
        mock_web_instance.assert_called_once()
        MockRag.return_value.assert_not_called()


if __name__ == "__main__":
    unittest.main()
