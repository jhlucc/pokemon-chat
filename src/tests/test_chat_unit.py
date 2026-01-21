import os
import unittest

if not os.getenv('RUN_INTEGRATION_TESTS'):
    raise unittest.SkipTest("Integration tests are skipped by default. Set RUN_INTEGRATION_TESTS=1 to run.")


import asyncio
import json
from src.agents.chat_agent import PokemonKGChatAgent, AgentState
from langchain_core.messages import HumanMessage

def test_chat_node():
    agent = PokemonKGChatAgent()
    
    # Mock state
    state = {
        "messages": [HumanMessage(content="Hello")],
        "thread_id": "test_thread",
        "user_id": "test_user",
        "response_mode": "json"
    }
    
    print("\n--- Testing _chat node with JSON mode (expecting error handling) ---")
    try:
        result = agent._chat(state)
        print("Node execution finished.")
        messages = result.get("messages", [])
        if messages:
            last_msg = messages[-1]
            print(f"Output Content: {last_msg.content}")
            
            # Verify if it returns a JSON with error info
            try:
                data = json.loads(last_msg.content)
                if "error_code" in data:
                    print("SUCCESS: Received structured ErrorResponse.")
                else:
                    print(f"Received JSON: {data}")
            except Exception as e:
                print(f"FAILURE: Output is not JSON: {e}")
                
    except Exception as e:
        print(f"Node execution failed with exception: {e}")

if __name__ == "__main__":
    test_chat_node()
