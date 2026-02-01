import re

from src.plugins.parser import DocxParser, ExcelParser, PdfParser, PptParser
from src.utils.logger import get_logger

_log = get_logger(__name__)


def _clean_spaced_text(text: str) -> str:
    """
    清理 PDF 解析产生的异常空格。
    某些 PDF 会将每个字符单独存储，导致 "H e l l o" 这样的输出。
    策略：多空格=词边界，单空格=字符间距。
    """
    if not text:
        return text

    lines = text.split("\n")
    cleaned_lines = []
    for line in lines:
        words = line.split()
        if not words:
            cleaned_lines.append(line)
            continue

        single_char_count = sum(1 for w in words if len(w) == 1 and w.isalpha())
        if len(words) > 5 and single_char_count / len(words) > 0.3:
            # 先保护多空格（词边界）为占位符
            cleaned = re.sub(r" {2,}", "\x00", line)
            # 移除单个空格（字符间距）
            cleaned = re.sub(r"(?<=[a-zA-Z]) (?=[a-zA-Z])", "", cleaned)
            # 恢复词边界空格
            cleaned = cleaned.replace("\x00", " ")
            cleaned_lines.append(cleaned)
        else:
            cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


class DeepDocParser:
    @staticmethod
    def parse(file_path: str) -> str:
        ext = file_path.split(".")[-1].lower()
        if not ext.startswith("."):
            ext = "." + ext

        parser = None

        if ext == ".pdf":
            parser = PdfParser()
            # DeepDoc PDF returns (text_block, table_block) tuples or strings
            text_blocks, _ = parser(file_path, need_image=False, zoomin=3, return_html=False)
        elif ext == ".docx":
            parser = DocxParser()
            text_blocks, _ = parser(file_path)
        elif ext in [".ppt", ".pptx"]:
            parser = PptParser()
            text_blocks = parser(file_path)
        elif ext in [".xls", ".xlsx", ".csv"]:
            parser = ExcelParser()
            text_blocks = parser(file_path)
        else:
            raise ValueError(f"DeepDocParser: Unsupported extension {ext}")

        # Normalize output: join blocks
        all_text = []
        # text_blocks can be list of str OR list of tuples
        # DeepDoc output varies by parser

        # Pdf/Docx return list of (text, bbox) or text string match
        # Excel/Ppt return list of strings usually

        for block in text_blocks:
            if isinstance(block, str):
                all_text.append(block)
            elif isinstance(block, tuple):
                # block[0] is typically text
                all_text.append(str(block[0]))

        # 清理异常空格并返回
        result = "\n".join(all_text)
        return _clean_spaced_text(result)
