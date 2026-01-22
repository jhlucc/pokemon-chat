
import unittest
from unittest.mock import patch, MagicMock
from src.knowledge.core.indexing import parse_file
import os

class TestParserRouting(unittest.TestCase):
    
    @patch('src.knowledge.ingestion.parsers.base.os.path.isfile', return_value=True)
    @patch('src.knowledge.ingestion.parsers.base.MarkItDownParser')
    @patch('src.knowledge.ingestion.parsers.base.DeepDocParser')
    def test_routing(self, MockDeepDoc, MockMarkItDown, MockIsFile):
        
        # --- Test PDF ---
        # Should route to DeepDocParser
        parse_file("test.pdf")
        MockDeepDoc.parse.assert_called_with("test.pdf")
        MockMarkItDown.parse.assert_not_called()
        
        MockDeepDoc.reset_mock()
        MockMarkItDown.reset_mock()
        
        # --- Test PPT ---
        # Should route to DeepDocParser
        parse_file("test.ppt")
        MockDeepDoc.parse.assert_called_with("test.ppt")
        MockMarkItDown.parse.assert_not_called()
        
        MockDeepDoc.reset_mock()
        MockMarkItDown.reset_mock()
        
        # --- Test DOCX ---
        # Should route to MarkItDownParser
        # Mock success
        MockMarkItDown.parse.return_value = "docx content"
        
        parse_file("test.docx")
        MockMarkItDown.parse.assert_called_with("test.docx")
        # DeepDoc not called if success
        MockDeepDoc.parse.assert_not_called()

        print("✅ All routing tests passed!")

if __name__ == "__main__":
    unittest.main()
