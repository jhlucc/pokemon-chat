
import unittest
from unittest.mock import patch, MagicMock
from src.knowledge.core.indexing import parse_file
import os

class TestParserRouting(unittest.TestCase):
    
    @patch('src.knowledge.core.indexing.os.path.isfile', return_value=True)
    @patch('src.knowledge.core.indexing.MarkItDown')
    @patch('src.knowledge.core.indexing.PdfParser')
    @patch('src.knowledge.core.indexing.PptParser')
    @patch('src.knowledge.core.indexing.DocxParser')
    def test_routing(self, MockDocx, MockPpt, MockPdf, MockMarkItDown, MockIsFile):
        # Configure Mock returns to satisfy unpacking
        mock_pdf_instance = MockPdf.return_value
        mock_pdf_instance.return_value = (["pdf content"], [])
        
        mock_ppt_instance = MockPpt.return_value
        mock_ppt_instance.return_value = ["ppt content"] # PptParser returns list, not tuple

        mock_docx_instance = MockDocx.return_value
        mock_docx_instance.return_value = (["docx content"], []) # DocxParser returns tuple

        # --- Test PDF ---
        parse_file("test.pdf")
        MockPdf.assert_called_once()
        MockMarkItDown.assert_not_called()
        
        # Reset
        MockPdf.reset_mock()
        MockMarkItDown.reset_mock()
        
        # --- Test PPT ---
        parse_file("test.ppt")
        MockPpt.assert_called_once()
        MockMarkItDown.assert_not_called()
        
        # Reset
        MockPpt.reset_mock()
        MockMarkItDown.reset_mock()
        
        # --- Test DOCX ---
        # Ensure MarkItDown works so we don't fallback (but if we do, Docx mock saves us)
        mock_md_instance = MockMarkItDown.return_value
        mock_md_instance.convert.return_value.text_content = "docx content"
        
        parse_file("test.docx")
        
        # Verify MarkItDown WAS called
        MockMarkItDown.assert_called_once()
        # Verify DocxParser was NOT called (success path)
        MockDocx.assert_not_called() 

        print("✅ All routing tests passed!")

if __name__ == "__main__":
    unittest.main()
