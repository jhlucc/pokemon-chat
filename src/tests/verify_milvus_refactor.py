import importlib


def _check_import(module: str, symbol: str) -> bool:
    try:
        mod = importlib.import_module(module)
        obj = getattr(mod, symbol)
        print(f"PASS: imported {module}.{symbol} -> {obj}")
        return True
    except Exception as e:
        print(f"FAIL: cannot import {module}.{symbol}: {e}")
        return False


def test_imports():
    print("Testing imports for MilvusService refactor...")
    ok = True

    ok &= _check_import("src.knowledge.vector.milvus_store", "MilvusService")
    ok &= _check_import("src.agents.tools.websearch.LiteWebSearcher", "WebSearcher")
    ok &= _check_import("src.agents.tools.websearch.TavilyWebSearcher", "IndustrialWebSearcher")

    if ok:
        print("All imports valid.")


if __name__ == "__main__":
    test_imports()
