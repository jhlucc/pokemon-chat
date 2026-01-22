
import sys
import os

sys.path.append(os.getcwd())

def test_legacy_config_removal():
    print("Testing legacy config removal...")
    
    # Test 1: Import src.config should fail
    try:
        import src.config
        print("❌ FAIL: src.config is still importable!")
    except ImportError:
        print("✅ PASS: src.config is correctly removed.")

    # Test 2: Import src.config.Config should fail
    try:
        from src.config import Config
        print("❌ FAIL: src.config.Config is still importable!")
    except ImportError:
        print("✅ PASS: src.config.Config is correctly removed.")
        
    # Test 3: Check src.config proxy in init (should fail or be gone)
    try:
        from src import config
        print("❌ FAIL: src.config proxy still exists in __init__.py")
    except ImportError:
        print("✅ PASS: src.config proxy is gone.")
        
    print("Verification complete.")

if __name__ == "__main__":
    test_legacy_config_removal()
