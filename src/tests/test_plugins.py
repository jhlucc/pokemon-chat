
from src.plugins import DocumentProcessorFactory, RapidOCRProcessor

def test_plugin_registry():
    print("\n--- Testing Plugin Registry ---")
    try:
        processor = DocumentProcessorFactory.get_processor("rapid_ocr")
        print(f"Successfully retrieved processor: {type(processor)}")
        assert isinstance(processor, RapidOCRProcessor)
        
        health = processor.check_health()
        print(f"Health check: {health}")
        
    except Exception as e:
        print(f"Plugin registry test failed: {e}")

if __name__ == "__main__":
    test_plugin_registry()
