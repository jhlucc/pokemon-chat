import os
from pathlib import Path

from src.utils.logger import get_logger

from .deepdoc_parser import DeepDocParser
from .markitdown_parser import MarkItDownParser

_log = get_logger(__name__)


def _is_garbled_text(text: str, threshold: float = 0.5) -> bool:
    """
    检测文本是否是乱码。
    通过检查有效字符（中文、英文、数字、常见标点）的比例来判断。
    同时检测是否有大量非常见 Unicode 字符（乱码特征）。
    """
    import re
    if not text or len(text.strip()) == 0:
        return True

    # 检查是否包含 CID 标记 (cid:xxx)
    if re.search(r'\(cid:\d+\)', text):
        return True

    non_space_text = re.sub(r'\s', '', text)
    if len(non_space_text) == 0:
        return True

    # 统计中文字符
    chinese_chars = re.findall(r'[\u4e00-\u9fff\u3400-\u4dbf]', text)

    # 统计基本 ASCII 可打印字符（英文、数字、常见标点）
    ascii_printable = re.findall(r'[a-zA-Z0-9,.!?;:\'"()\[\]{}<>/\\@#$%^&*+=_|~`\-]', text)

    # 统计中文标点
    chinese_punct = re.findall(r'[，。！？、；：""''（）【】《》—…·]', text)

    # 计算有效字符总数
    valid_count = len(chinese_chars) + len(ascii_printable) + len(chinese_punct)
    valid_ratio = valid_count / len(non_space_text)

    # 检测乱码特征：大量 Latin Extended / 特殊 Unicode 字符
    # 这些字符在正常中文或英文文本中很少出现
    weird_chars = re.findall(r'[\u0080-\u00ff\u0100-\u017f\u0180-\u024f]', text)
    weird_ratio = len(weird_chars) / len(non_space_text) if len(non_space_text) > 0 else 0

    # 如果有效字符比例低于阈值，或者有大量异常字符，判定为乱码
    if valid_ratio < threshold:
        return True

    # 如果异常字符超过 20%，很可能是乱码
    if weird_ratio > 0.2:
        return True

    return False


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
        # Keep OCR optional: the OCR stack (rapidocr/onnxruntime) is heavy and may be
        # intentionally absent in minimal deployments. Fallback to DeepDoc when missing.
        try:
            from src.plugins.vision._ocr import OCRHandler2  # noqa: WPS433 (lazy import)
        except Exception as e:
            _log.warning(f"parse_file - OCR backend unavailable ({e}); falling back to DeepDoc for {file_path}")
        else:
            ocr_handler = OCRHandler2(det_threshold=ocr_det_threshold)
            return ocr_handler.pdf_ocr_pipeline(file_path)

    # 2. Routing Logic
    # Preference:
    # PDF/PPT -> DeepDoc (Better layout analysis)
    # DOCX/Excel/TXT -> MarkItDown (Better Markdown conversion)

    # PDFs: prefer a lightweight text-layer extractor for fast, dependency-light parsing.
    # (OCR is handled above when `do_ocr=True` and optional OCR backends are available.)
    if ext == ".pdf":
        _log.info(f"parse_file - Using PyPDFLoader for {file_path}")
        try:
            from langchain_community.document_loaders import PyPDFLoader

            docs = PyPDFLoader(file_path).load()
            text = "\n\n".join(d.page_content for d in docs)

            # 检查是否是乱码，如果是则尝试 OCR
            if _is_garbled_text(text):
                _log.warning(f"PyPDFLoader 提取的文本疑似乱码，尝试 OCR: {file_path}")
                try:
                    from src.plugins.vision._ocr import OCRHandler2
                    ocr_handler = OCRHandler2(det_threshold=ocr_det_threshold)
                    return ocr_handler.pdf_ocr_pipeline(file_path)
                except Exception as ocr_e:
                    _log.warning(f"OCR 也失败了 ({ocr_e})，返回原始文本")
                    return text

            return text
        except Exception as e:
            _log.warning(f"PyPDFLoader failed for {file_path}: {e}. Falling back to MarkItDown.")
            return MarkItDownParser.parse(file_path)

    # PPT/PPTX: DeepDoc handles layout better.
    if ext in [".ppt", ".pptx"]:
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
