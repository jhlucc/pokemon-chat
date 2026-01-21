import os
try:
    from rapidocr_onnxruntime import RapidOCR
except ImportError:
    RapidOCR = None

from src.plugins.document_processor_base import BaseDocumentProcessor, ProcessingResult
from typing import Dict, Any

class RapidOCRProcessor(BaseDocumentProcessor):
    """RapidOCR 处理器"""
    
    def __init__(self):
        if RapidOCR is None:
            self.model = None
            self.error = "rapidocr_onnxruntime not installed"
        else:
            try:
                self.model = RapidOCR()
                self.error = None
            except Exception as e:
                self.model = None
                self.error = str(e)
                
    def process_file(self, file_path: str, params: Dict[str, Any] = None) -> ProcessingResult:
        if self.model is None:
            return ProcessingResult(content="", error=f"RapidOCR model not available: {self.error}")
        
        if not os.path.exists(file_path):
            return ProcessingResult(content="", error=f"File not found: {file_path}")
            
        try:
            # RapidOCR call
            result, _ = self.model(file_path)
            if not result:
                return ProcessingResult(content="")
            
            # Extract text
            # result format: [[[[x1,y1],[x2,y2],[x3,y3],[x4,y4]], text, confidence], ...]
            text_content = "\n".join([line[1] for line in result])
            
            return ProcessingResult(content=text_content, metadata={"engine": "rapidocr"})
            
        except Exception as e:
            return ProcessingResult(content="", error=f"OCR processing failed: {e}")

    def check_health(self) -> Dict[str, Any]:
        return {
            "available": self.model is not None,
            "error": self.error
        }
