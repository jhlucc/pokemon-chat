import os
import unittest

from langgraph.checkpoint.memory import MemorySaver

from src.agents.deep_agent import DeepAgent

# This is intentionally integration-flavoured (depends on optional components / model setup).
if not os.getenv("RUN_INTEGRATION_TESTS"):
    raise unittest.SkipTest("Integration tests are skipped by default. Set RUN_INTEGRATION_TESTS=1 to run.")


class TestDeepAgent(DeepAgent):
    def _build_graph(self):
        # Override to add a checkpointer required by aget_state() paths.
        from langgraph.graph import END, START, StateGraph

        from src.agents.deep_agent.context import DeepContext
        from src.agents.deep_agent.graph import critique_node, finalize_node, research_node, route_step

        builder = StateGraph(DeepContext)
        builder.add_node("research", research_node)
        builder.add_node("critique", critique_node)
        builder.add_node("finalize", finalize_node)
        builder.add_edge(START, "research")
        builder.add_conditional_edges("research", route_step, {"critique": "critique", "finalize": "finalize"})
        builder.add_edge("critique", "research")
        builder.add_edge("finalize", END)

        return builder.compile(checkpointer=MemorySaver())


async def test_deep_agent_query_smoke():
    agent = TestDeepAgent()
    response = await agent.query("Test Topic", thread_id="test_deep")
    assert isinstance(response, str)
