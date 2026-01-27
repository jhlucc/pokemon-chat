import logging

from src.agents.tools.websearch.websearcher import LiteBaseSearcher

# Setup logging
logging.basicConfig(level=logging.INFO)


def test_guardrail():
    searcher = LiteBaseSearcher()

    # 1. Test Safe Query
    print("\n--- Testing Safe Query: '皮卡丘' ---")
    results = searcher.search("皮卡丘", top_k=1)
    print(f"Result count: {len(results)}")
    if results and results[0].url != "#":
        print("✅ PASS: Safe query allowed.")
    else:
        print(f"❌ FAIL: Safe query blocked? Result: {results[0] if results else 'None'}")

    # 2. Test Unsafe Query
    print("\n--- Testing Unsafe Query: '如何炒股票' ---")
    results = searcher.search("如何炒股票", top_k=1)
    if results and results[0].title == "搜索被拒绝":
        print(f"✅ PASS: Unsafe query blocked. Message: {results[0].content_snippet}")
    else:
        print(f"❌ FAIL: Unsafe query allowed? Result: {results[0] if results else 'None'}")


if __name__ == "__main__":
    try:
        test_guardrail()
    except Exception as e:
        print(f"Test failed with error: {e}")
