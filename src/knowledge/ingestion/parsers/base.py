import os
from pathlib import Path

from src.plugins.vision._ocr import OCRHandler2
from src.utils.logger import get_logger

from .deepdoc_parser import DeepDocParser
from .markitdown_parser import MarkItDownParser

_log = get_logger(__name__)


def parse_file(
    file_path: str,
    do_ocr: bool = False,
    ocr_det_threshold: float = 0.3,
) -> str:
    """
    Unified file parser entry point.
    Routes to DeepDoc, MarkItDown, or OCR based on file type and config.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"parse_file: file not found => {file_path}")

    ext = Path(file_path).suffix.lower()

    # 1. ORC specific overriding
    if ext == ".pdf" and do_ocr:
        _log.info(f"parse_file - Using OCR pipeline for {file_path}")
        ocr_handler = OCRHandler2(det_threshold=ocr_det_threshold)
        return ocr_handler.pdf_ocr_pipeline(file_path)

    # 2. Routing Logic
    # Preference:
    # PDF/PPT -> DeepDoc (Better layout analysis)
    # DOCX/Excel/TXT -> MarkItDown (Better Markdown conversion)

    if ext in [".pdf", ".ppt", ".pptx"]:
        # DeepDoc Strategy
        _log.info(f"parse_file - Using DeepDoc for {file_path}")
        return DeepDocParser.parse(file_path)

    elif ext in [".docx", ".xls", ".xlsx", ".csv", ".txt", ".md"]:
        # MarkItDown Strategy with Fallback
        try:
            _log.info(f"parse_file - Using MarkItDown for {file_path}")
            return MarkItDownParser.parse(file_path)
        except Exception as e:
            _log.warning(f"MarkItDown failed for {file_path}: {e}. Falling back to DeepDoc.")
            # Fallback for DOCX/Excel to DeepDoc parsers if available
            if ext in [".docx", ".xls", ".xlsx", ".csv"]:
                return DeepDocParser.parse(file_path)
            elif ext in [".txt", ".md"]:
                # Simple read fallback
                with open(file_path, encoding="utf-8") as f:
                    return f.read()
            raise e

    raise ValueError(f"parse_file: Unsupported file type => {ext}")
