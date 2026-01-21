import os
import unittest

if not os.getenv('RUN_INTEGRATION_TESTS'):
    raise unittest.SkipTest("Integration tests are skipped by default. Set RUN_INTEGRATION_TESTS=1 to run.")


import asyncio
from src.agents.chat_agent import PokemonKGChatAgent
from langgraph.types import Command
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnableLambda

async def test_interrupt_flow():
    agent = PokemonKGChatAgent()
    thread_id = "test_interrupt_thread"
    config = {"configurable": {"thread_id": thread_id}}
    
    # MOCK LLM to force routing to 'approval'
    # The supervisor uses: chain = prompt | self.llm | JsonOutputParser()
    # We can mock self.llm.invoke to return a JSON string?
    # No, self.llm is a RunnableLambda wrapping middleware.
    # But supervisor calls chain.invoke({...}) which uses self.llm.
    # Let's mock agent.llm.invoke
    
    def mock_llm_call(input_val):
        # input_val is a list of messages or prompt value
        # Supervisor expects the chain to return a dict (JsonOutputParser)
        # But wait, the chain is: prompt | self.llm | JsonOutputParser()
        # So self.llm must return an AIMessage with JSON content.
        return AIMessage(content='{"next": "approval"}')

    agent.llm = RunnableLambda(mock_llm_call)
    
    initial_state = {
        "messages": [HumanMessage(content="Trigger approval")],
        "thread_id": thread_id
    }
    
    print("\n--- 1. Starting execution (Expect Interrupt) ---")
    
    # We need to catch the interrupt if it propagates, OR just check if it stops.
    # LangGraph usually returns to caller when interrupt happens.
    
    await agent.graph.ainvoke(initial_state, config)
    
    # Check state
    snapshot = await agent.graph.aget_state(config)
    print(f"Current Node: {snapshot.next}")
    if snapshot.tasks:
        print(f"Pending Tasks: {len(snapshot.tasks)}")
        if snapshot.tasks[0].interrupts:
            print(f"Interrupts found: {snapshot.tasks[0].interrupts}")
            
    # 2. Resume execution
    print("\n--- 2. Resuming execution with 'yes' ---")
    try:
        # Resume with "yes"
        # The interrupt returns the value given in resume
        await agent.graph.ainvoke(Command(resume="yes"), config)
        print("Resumed successfully.")
        
        snapshot = await agent.graph.aget_state(config)
        print(f"Final State Values: {snapshot.values.get('approval_status')}")
        print(f"Final State Feedback: {snapshot.values.get('user_feedback')}")
        
    except Exception as e:
        print(f"Resume failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_interrupt_flow())
