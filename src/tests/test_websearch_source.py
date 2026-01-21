from src.agents.tools.websearch.websearcher import LiteBaseSearcher
from src.models.schemas import Source
from src.core.settings import settings

async def test_search():
    searcher = LiteBaseSearcher()
    # Mocking search_and_generate behavior as it's the main entry point logic inside chat_agent (simulated)
    # But here we want to test .search().
    # Since LiteBaseSearcher calls utils.search which is async, let's see if we can run it.
    # Note: utils.search requires web access. If no access, it returns empty or error.
    # We can inspect the code to see if it returns Source objects now.
    
    # Actually, LiteBaseSearcher.search is synchronous.
    try:
        results = searcher.search("Pikachu", top_k=1)
        print(f"Results type: {type(results)}")
        if results:
            print(f"First result type: {type(results[0])}")
            if isinstance(results[0], Source):
                print("SUCCESS: Result is a Source object.")
                print(results[0].model_dump_json(indent=2))
            else:
                print("FAILURE: Result is not a Source object.")
        else:
            print("No results found (network might be offline), but no error.")
            
    except Exception as e:
        print(f"Search failed: {e}")

if __name__ == "__main__":
    # Settings needs to be initialized
    print(f"Web Search Enabled: {settings.features.enable_web_search}")
    
    # Run test
    test_search()
