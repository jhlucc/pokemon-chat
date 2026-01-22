
import os
import pandas as pd
from src.knowledge.core.indexing import parse_file

def test_markitdown_parsing():
    # Create a dummy excel file
    test_file = "test_data.xlsx"
    df = pd.DataFrame({"Name": ["Pikachu", "Charizard"], "Type": ["Electric", "Fire"]})
    df.to_excel(test_file, index=False)
    
    try:
        print(f"Testing parsing for {test_file}...")
        text = parse_file(test_file)
        print("--- Extracted Text ---")
        print(text)
        print("----------------------")
        
        if "Pikachu" in text and "Charizard" in text:
            print("✅ PASS: MarkItDown parsing successful (content found).")
        else:
            print("❌ FAIL: Content missing.")
            
    except Exception as e:
        print(f"❌ FAIL: Parsing failed with error: {e}")
    finally:
        if os.path.exists(test_file):
            os.remove(test_file)

if __name__ == "__main__":
    test_markitdown_parsing()
