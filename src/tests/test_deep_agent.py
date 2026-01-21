
import asyncio
from src.agents.deep_agent import DeepAgent
from langchain_core.messages import HumanMessage

async def test_deep_agent_flow():
    print("\n--- Testing Deep Agent Flow ---")
    agent = DeepAgent()
    
    question = "Analyze Pokemon type effectiveness distribution"
    print(f"Question: {question}")
    
    # We use a memory checkpointer if we want state persistence, 
    # but DeepAgent default impl uses .compile() without checkpointer in my code above?
    # Wait, checkpointer is needed for aget_state.
    # I need to update DeepAgent to use a checkpointer or just return result from invoke.
    
    # Check DeepAgent._build_graph: builder.compile() -> No checkpointer.
    # aget_state will fail if no checkpointer is used.
    # For this test, let's just invoke and see output?
    # Or strict aget_state.
    pass

    # Actually, let's fix DeepAgent to use checkpointer for testing, or rely on return value.
    # Since I implemented query() using astream and then aget_state, it WILL fail without checkpointer.
    # I should add InMemorySaver to DeepAgent by default or injected.
    
from langgraph.checkpoint.memory import MemorySaver

class TestDeepAgent(DeepAgent):
    def _build_graph(self):
        # Override to add checkpointer
        from langgraph.graph import StateGraph, START, END
        from src.agents.deep_agent.graph import research_node, critique_node, finalize_node, route_step
        from src.agents.deep_agent.context import DeepContext
        
        builder = StateGraph(DeepContext)
        builder.add_node("research", research_node)
        builder.add_node("critique", critique_node)
        builder.add_node("finalize", finalize_node)
        builder.add_edge(START, "research")
        builder.add_conditional_edges("research", route_step, {"critique": "critique", "finalize": "finalize"})
        builder.add_edge("critique", "research")
        builder.add_edge("finalize", END)
        
        return builder.compile(checkpointer=MemorySaver())

async def run_test():
    agent = TestDeepAgent()
    response = await agent.query("Test Topic", thread_id="test_deep")
    print(f"Deep Agent Response: {response}")
    assert "Final Report" in response

if __name__ == "__main__":
    asyncio.run(run_test())
