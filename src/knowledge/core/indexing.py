import os
from pathlib import Path
from typing import List, Optional
from src.plugins.parser import PdfParser, DocxParser, ExcelParser, PptParser, TxtParser
from src.plugins.vision._ocr import OCRHandler2
from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter
from src.utils.logger import get_logger

from markitdown import MarkItDown

_log = get_logger(__name__)


from src.knowledge.ingestion.parsers.base import parse_file as new_parse_file

def parse_file(
        file_path: str,
        do_ocr: bool = False,
        ocr_det_threshold: float = 0.3,
) -> str:
    """
    Deprecated: Delegates to src.knowledge.ingestion.parsers.base.parse_file
    """
    return new_parse_file(file_path, do_ocr, ocr_det_threshold)


def chunk_file(
        file_path: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
        do_ocr: bool = False,
        ocr_det_threshold: float = 0.3,
) -> List[Document]:
    # 先将文件解析成纯文本
    text = parse_file(file_path, do_ocr=do_ocr, ocr_det_threshold=ocr_det_threshold)
    # 创建一个文本切分器，这里示例用 CharacterTextSplitter + tiktoken encoder
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    sub_texts = splitter.split_text(text)
    docs = []
    for idx, s in enumerate(sub_texts):
        metadata = {
            "source_file": file_path,
            "chunk_index": idx,
        }
        docs.append(Document(page_content=s, metadata=metadata))

    return docs


def chunk_text(
        text: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
) -> List[Document]:
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    sub_texts = splitter.split_text(text)

    return [Document(page_content=st, metadata={"type": "inline_text"}) for st in sub_texts]


if __name__ == "__main__":
    test_pdf = "C:/Users/luke/Desktop/Smart-Assistant/deepdoc/data/identity.txt"

    # 1) 不需要OCR的方式解析 + 切分
    doc_chunks = chunk_file(test_pdf, chunk_size=100, chunk_overlap=50, do_ocr=False)
    print(f"Got {len(doc_chunks)} chunks from normal PDF parsing.")
    for c in doc_chunks[:12]:
        one_line_text = c.page_content.replace("\n", "").replace("\r", "")
        print("CHUNK:", one_line_text, "...")
        print("------------------")
    # doc_chunks_ocr = chunk_file(test_pdf, chunk_size=800, do_ocr=True)
    # print(f"Got {len(doc_chunks_ocr)} chunks from OCR PDF pipeline.")
    # for c in doc_chunks_ocr[:12]:
    #     one_line_text = c.page_content.replace("\n", "").replace("\r", "")
    #     print("CHUNK:", one_line_text, "...")
    #     print("------------------")
