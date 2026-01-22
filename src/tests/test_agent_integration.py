import unittest
from src.agents.manager import agent_manager
from langchain_core.messages import HumanMessage

class TestAgentIntegration(unittest.TestCase):
    def test_supervisor_agent_retrieval(self):
        agent = agent_manager.get_agent("supervisor_agent")
        self.assertIsNotNone(agent)
        info = agent.get_info()
        self.assertEqual(info["name"], "supervisor_agent")
        
    def test_graph_compilation(self):
        agent = agent_manager.get_agent("supervisor_agent")
        # Check if graph is built (accessing property triggers build)
        self.assertIsNotNone(agent.graph)

if __name__ == "__main__":
    unittest.main()
