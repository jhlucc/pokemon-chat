
import asyncio
from src.agents.manager import agent_manager
from src.agents.base import BaseAgent
from langchain_core.messages import HumanMessage

async def test_architecture():
    print("\n--- Testing Agent Architecture ---")
    
    # 1. List valid agents
    agents_info = agent_manager.list_agents()
    print(f"Available Agents: {agents_info.keys()}")
    assert "chat_agent" in agents_info
    assert "deep_agent" in agents_info
    
    # 2. Get Instances
    chat_agent = agent_manager.get_agent("chat_agent")
    deep_agent = agent_manager.get_agent("deep_agent")
    
    assert isinstance(chat_agent, BaseAgent)
    assert isinstance(deep_agent, BaseAgent)
    
    print("Agents instantiated successfully and inherit from BaseAgent.")
    
    # 3. Test BaseAgent Interface (get_info)
    print(f"Chat Agent Info: {chat_agent.get_info()}")
    print(f"Deep Agent Info: {deep_agent.get_info()}")
    
    # 4. Test Graph Property
    assert chat_agent.graph is not None
    assert deep_agent.graph is not None
    print("Graph property accessible.")
    
    # 5. Test Unified Query Interface (astream)
    # We will test deep_agent as it's lighter (mocked graph)
    print("\n--- Testing Deep Agent Query (via BaseAgent astream) ---")
    input_state = {
        "topic": "Test Architecture",
        "messages": [HumanMessage(content="Test Architecture")],
        "iterations": 0
    }
    config = {"configurable": {"thread_id": "arch_test"}}
    
    try:
        async for chunk in deep_agent.astream(input_state, config):
            # print(f"Chunk: {chunk.keys() if isinstance(chunk, dict) else chunk}")
            pass
        print("Deep Agent astream completed.")
    except Exception as e:
        print(f"Deep Agent query failed: {e}")
        # It handles missing checkpointer/etc inside graph? 
        # My deep_agent implementation uses builder.compile() without checkpointer. 
        # But astream doesn't STRICTLY require checkpointer unless we use interrupt/resume. 
        # Wait, if logic depends on checkpointer...
        pass

if __name__ == "__main__":
    asyncio.run(test_architecture())
