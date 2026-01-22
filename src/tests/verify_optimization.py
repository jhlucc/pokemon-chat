import unittest
from unittest.mock import MagicMock, patch
import asyncio
import threading
import time
from langchain_core.messages import HumanMessage, AIMessage

from src.agents.middleware.long_term_memory import LongTermMemoryMiddleware
from src.agents.middleware.base import MiddlewareContext
from src.agents.chat_agent import PokemonKGChatAgent
from src.agents.deep_agent.graph import DeepAgent

class TestOptimization(unittest.TestCase):
    def test_memory_background_summarization(self):
        print("\n--- Testing Memory Background Summarization ---")
        
        # Mock Memory Manager
        mock_manager = MagicMock()
        middleware = LongTermMemoryMiddleware(memory_manager=mock_manager)
        
        # Mock LLM inside the background method (using patch)
        with patch('langchain_openai.ChatOpenAI') as MockLLM:
            mock_llm_instance = MockLLM.return_value
            mock_llm_instance.invoke.return_value = AIMessage(content="User likes Pikachu")
            
            # Prepare state and context
            state = {
                "messages": [
                    HumanMessage(content="I love Pikachu"),
                    AIMessage(content="Pikachu is cute"),
                    HumanMessage(content="He is electric type"),
                    AIMessage(content="Yes, indeed")
                ]
            }
            context = MiddlewareContext(agent_name="test_agent", user_id="u1", thread_id="t1")
            
            # Run middleware hook
            # It should start a thread
            asyncio.run(middleware.after_agent(state, context))
            
            # Wait a bit for thread to run
            time.sleep(0.5)
            
            # Verify LLM was called
            # Since it's in a thread, we check if mock_llm_instance was used
            mock_llm_instance.invoke.assert_called()
            print("LLM invoked for summarization.")
            
            # Verify memory manager added memory
            mock_manager.add_memory_sync.assert_called_with(
                "User likes Pikachu", 
                metadata={"user_id": "u1", "thread_id": "t1", "source": "test_agent_summary", "timestamp": "t1"}
            )
            print("Memory added successfully.")

    def test_deep_agent_integration(self):
        print("\n--- Testing Deep Agent Integration ---")
        
        # We need to mock settings to avoid real init issues if env not set
        with patch('src.agents.chat_agent.settings') as mock_settings:
            mock_settings.features.enable_knowledge_graph = False
            mock_settings.features.enable_web_search = False
            # Fix checkpointer mock
            mock_settings.agent.checkpointer_type.lower.return_value = "memory"
            mock_settings.agent.conversation_max_messages = 10
            mock_settings.llm.model_name = "test-model"
            mock_settings.llm.api_base = "http://mock"
            mock_settings.llm.api_key = "mock"
            
            # Mock other inits
            with patch('src.agents.chat_agent.PokemonLightRAG'), \
                 patch('src.agents.chat_agent.LiteBaseSearcher'), \
                 patch('src.agents.chat_agent.PokemonStatsAgent'), \
                 patch('src.agents.chat_agent.PokedexAgent'), \
                 patch('src.agents.chat_agent.TrainerAgent'), \
                 patch('src.agents.chat_agent.DeepAgent') as MockDeepAgent:
                
                agent = PokemonKGChatAgent()
                
                # Check members
                print(f"Members: {agent.members}")
                self.assertIn("deep_researcher", agent.members)
                
                # Check graph nodes
                nodes = agent.graph.nodes
                print(f"Graph Nodes: {nodes.keys()}")
                self.assertIn("deep_researcher", nodes)
                
                print("Deep Agent integrated into Chat Agent graph.")

if __name__ == "__main__":
    unittest.main()
