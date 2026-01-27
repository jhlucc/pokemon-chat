import importlib


def test_legacy_config_removal():
    print("Testing legacy config removal...")

    # Test 1: importing src.config should fail
    try:
        importlib.import_module("src.config")
        print("FAIL: src.config is still importable!")
    except ImportError:
        print("PASS: src.config is correctly removed.")

    # Test 2: importing src.config.Config should fail
    try:
        mod = importlib.import_module("src.config")
        _ = mod.Config
        print("FAIL: src.config.Config is still importable!")
    except (ImportError, AttributeError):
        print("PASS: src.config.Config is correctly removed.")

    # Test 3: src.__init__ should not provide a legacy `config` proxy
    try:
        src_pkg = importlib.import_module("src")
        _ = src_pkg.config
        print("FAIL: src.config proxy still exists in src/__init__.py")
    except AttributeError:
        print("PASS: src.config proxy is gone.")

    print("Verification complete.")


if __name__ == "__main__":
    test_legacy_config_removal()
