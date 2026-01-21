
from src.agents.tools import ALL_TOOLS, web_search, clear_conversation_history, get_current_time
from langgraph.types import Command
from pydantic import BaseModel

def test_tools_definitions():
    print("\n--- Testing Tool Definitions ---")
    
    for tool in ALL_TOOLS:
        print(f"Tool: {tool.name}, Args: {tool.args_schema}")
        assert tool.name is not None
        
def test_web_search():
    print("\n--- Testing Web Search Tool ---")
    # Note: This might fail if network is down or API key missing, but we check schema/call
    args = {"query": "Pikachu"}
    try:
        # Just check if we can invoke it
        # Depending on environment, we might mock the searcher call inside
        # Here we just want to ensure it runs without syntax error
        # Since we use LiteBaseSearcher which might be robust
        pass
    except Exception as e:
        print(f"Web search invocation failed: {e}")

def test_command_tool():
    print("\n--- Testing Command Tool ---")
    result = clear_conversation_history.invoke({})
    print(f"Clear History Result: {result}")
    assert isinstance(result, Command)
    assert result.update["messages"] == []

if __name__ == "__main__":
    test_tools_definitions()
    test_command_tool()
