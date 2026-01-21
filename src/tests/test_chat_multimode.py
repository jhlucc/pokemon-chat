import os
import unittest

if not os.getenv('RUN_INTEGRATION_TESTS'):
    raise unittest.SkipTest("Integration tests are skipped by default. Set RUN_INTEGRATION_TESTS=1 to run.")


import asyncio
import json
from src.agents.chat_agent import PokemonKGChatAgent
from src.models.schemas import AgentResponse

async def test_multimode():
    agent = PokemonKGChatAgent()
    
    # 1. Test Text Mode (Default)
    print("\n--- Testing Text Mode ---")
    question = "Who is Pikachu?"
    print(f"Question: {question}")
    async for chunk in agent.query(question, meta={"response_mode": "text"}):
        print(f"Output chunk: {chunk}")
        assert isinstance(chunk, str)

    # 2. Test JSON Mode
    print("\n--- Testing JSON Mode ---")
    question = "Who is Pikachu?"
    print(f"Question: {question}")
    
    full_output = ""
    async for chunk in agent.query(question, meta={"response_mode": "json"}):
        full_output += chunk
    
    print(f"Full Output: {full_output}")
    
    try:
        data = json.loads(full_output)
        print("JSON parse successful.")
        
        # Validate against schema
        if "error_code" in data:
            print(f"Received ErrorResponse: {data}")
        else:
            response = AgentResponse(**data)
            print("AgentResponse validation successful.")
            print(f"Content: {response.content}")
            print(f"Confidence: {response.confidence}")
            
    except json.JSONDecodeError as e:
        print(f"JSON decode failed: {e}")
    except Exception as e:
        print(f"Validation failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_multimode())
