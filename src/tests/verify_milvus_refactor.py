
import sys
import os

# Ensure src is in python path
sys.path.append(os.getcwd())

def test_imports():
    print("Testing imports for MilvusService refactor...")
    try:
        from src.knowledge.vector.milvus_store import MilvusService
        print("✅ Successfully imported MilvusService from new location.")
    except ImportError as e:
        print(f"❌ Failed to import MilvusService from new location: {e}")
        return

    try:
        from src.agents.tools.websearch.LiteWebSearcher import WebSearcher
        print("✅ Successfully imported WebSearcher (LiteWebSearcher).")
    except ImportError as e:
        print(f"❌ Failed to import WebSearcher: {e}")
        return

    try:
        from src.agents.tools.websearch.TavilyWebSearcher import IndustrialWebSearcher
        print("✅ Successfully imported IndustrialWebSearcher (TavilyWebSearcher).")
    except ImportError as e:
        print(f"❌ Failed to import IndustrialWebSearcher: {e}")
        return

    print("All imports valid.")

if __name__ == "__main__":
    test_imports()
