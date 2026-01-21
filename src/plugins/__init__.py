from .document_processor_base import BaseDocumentProcessor, ProcessingResult
from .document_processor_factory import DocumentProcessorFactory
from .rapid_ocr_processor import RapidOCRProcessor

# Register plugins
DocumentProcessorFactory.register("rapid_ocr", RapidOCRProcessor)

__all__ = [
    "BaseDocumentProcessor",
    "ProcessingResult",
    "DocumentProcessorFactory",
    "RapidOCRProcessor"
]
