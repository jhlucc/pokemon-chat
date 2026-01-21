import os
import unittest

if not os.getenv('RUN_INTEGRATION_TESTS'):
    raise unittest.SkipTest("Integration tests are skipped by default. Set RUN_INTEGRATION_TESTS=1 to run.")


import pytest
from langchain_core.messages import HumanMessage, SystemMessage
from src.agents.middleware.injection import InjectionMiddleware
from src.agents.middleware.base import MiddlewareContext
from src.agents.context.prompts import dynamic_prompt, DynamicPromptRegistry, state_aware_prompt

# 1. Test Injection Middleware
def test_injection_middleware():
    print("\n--- Testing Injection Middleware ---")
    
    def simple_injector(context):
        return SystemMessage(content="[INJECTED] Context Info")
        
    middleware = InjectionMiddleware(injectors=[simple_injector])
    
    messages = [
        SystemMessage(content="Original System"),
        HumanMessage(content="Hello")
    ]
    context = MiddlewareContext(agent_name="test")
    
    processed = middleware.before_model(messages, context)
    
    for m in processed:
        print(f"[{type(m).__name__}] {m.content}")
        
    # Check assertions
    assert len(processed) == 3
    # Logic in middleware puts injected system messages... 
    # Current implementation: [Original System] + [New System] + ...
    assert "Original System" in processed[0].content
    assert "[INJECTED]" in processed[1].content 

# 2. Test Dynamic Prompt
def test_dynamic_prompt():
    print("\n--- Testing Dynamic Prompt ---")
    
    # Test built-in sample
    prompt_func = DynamicPromptRegistry.get("state_aware")
    assert prompt_func is not None
    
    # Short history
    base = "You are a bot."
    res1 = prompt_func({"messages": [1, 2]}, base)
    print(f"Short history result: {res1}")
    assert res1 == base
    
    # Long history
    res2 = prompt_func({"messages": [i for i in range(25)]}, base)
    print(f"Long history result: {res2}")
    assert "对话历史较长" in res2

if __name__ == "__main__":
    test_injection_middleware()
    test_dynamic_prompt()
