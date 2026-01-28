from .document_processor_base import BaseDocumentProcessor, ProcessingResult
from .document_processor_factory import DocumentProcessorFactory

#
# Optional plugins
# -------------
# Some processors (e.g. RapidOCR) pull in heavy optional dependencies.
# Keep imports resilient so the rest of the system (parsers/tools) works even
# when optional OCR stacks are not installed.
#
try:
    from .rapid_ocr_processor import RapidOCRProcessor  # noqa: F401
except Exception:
    RapidOCRProcessor = None  # type: ignore[assignment]
else:
    # Register plugins only when import succeeds.
    DocumentProcessorFactory.register("rapid_ocr", RapidOCRProcessor)

__all__ = ["BaseDocumentProcessor", "ProcessingResult", "DocumentProcessorFactory"]
if RapidOCRProcessor is not None:
    __all__.append("RapidOCRProcessor")
