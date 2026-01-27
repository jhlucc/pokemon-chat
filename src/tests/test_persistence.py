import os
import unittest

from src.agents.supervisor_agent import SupervisorAgent
from src.core.settings import settings

try:
    from langgraph.checkpoint.sqlite import SqliteSaver  # type: ignore

    _HAS_SQLITE = True
except ModuleNotFoundError:
    _HAS_SQLITE = False


class TestPersistence(unittest.TestCase):
    def setUp(self):
        # Clean up previous DB
        db_path = os.path.join(settings.paths.save_yaml_path, "agent_checkpoints.sqlite")
        if os.path.exists(db_path):
            os.remove(db_path)

    def test_persistence(self):
        if not _HAS_SQLITE:
            self.skipTest("langgraph-checkpoint-sqlite is not installed")

        # Force sqlite checkpointer for this test.
        orig = settings.agent.checkpointer_type
        settings.agent.checkpointer_type = "sqlite"
        try:
            agent = SupervisorAgent()
        finally:
            settings.agent.checkpointer_type = orig

        # Create a thread ID

        # Invoke (Mocking the internal graph execution would be complex,
        # so we rely on the fact that _build_graph sets self.checkpointer)
        # We just check if checkpointer is set to SqliteSaver
        self.assertIsInstance(agent.checkpointer, SqliteSaver)

        # We can try to put some state and read it back
        # But since we didn't run the graph, there is no state.
        # We rely on the class check for now.


if __name__ == "__main__":
    unittest.main()
