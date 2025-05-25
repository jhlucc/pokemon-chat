import os
from pathlib import Path
from typing import List, Optional
from deepdoc.parser import PdfParser, DocxParser, ExcelParser, PptParser, TxtParser
from deepdoc.vision._ocr import OCRHandler2
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from src.utils import logger

_log = logger.LogManager()


def parse_file(
        file_path: str,
        do_ocr: bool = False,
        ocr_det_threshold: float = 0.3,
) -> str:
    """
    根据文件后缀来调用deepdoc对应的解析器，或 OCR。
    返回合并后的纯文本（字符串）。
    do_ocr为 True 时，优先使用 OCRHandler2.pdf_ocr_pipeline() 或其他 OCR 流程。
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"parse_file: file not found => {file_path}")
    ext = Path(file_path).suffix.lower()

    if ext == ".pdf":
        if do_ocr:
            _log.info(f"parse_file - Using OCR pipeline for {file_path}")
            ocr_handler = OCRHandler2(det_threshold=ocr_det_threshold)
            return ocr_handler.pdf_ocr_pipeline(file_path)

        _log.info(f"parse_file - Using PdfParser for {file_path}")
        parser = PdfParser()
        text_blocks, _tbls_or_figs = parser(file_path, need_image=False, zoomin=3, return_html=False)
        all_text = []
        for block in text_blocks:
            if isinstance(block, str):
                all_text.append(block)
            elif isinstance(block, tuple):
                all_text.append(block[0])
        return "\n".join(all_text)

    elif ext == ".docx":
        _log.info(f"parse_file - Using DocxParser for {file_path}")
        parser = DocxParser()
        text_blocks, _tbls_or_figs = parser(file_path)
        all_text = []
        for block in text_blocks:
            if isinstance(block, str):
                all_text.append(block)
            elif isinstance(block, tuple):
                all_text.append(block[0])
        return "\n".join(all_text)
    elif ext in [".txt", ".md"]:
        _log.info(f"parse_file - Using simple read for {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    elif ext in [".xls", ".xlsx", ".csv"]:
        _log.info(f"parse_file - Using ExcelParser for {file_path}")
        parser = ExcelParser()
        text_blocks = parser(file_path)
        return "\n".join(text_blocks)

    elif ext in [".ppt", ".pptx"]:
        _log.info(f"parse_file - Using PptParser for {file_path}")
        parser = PptParser()
        text_blocks = parser(file_path)
        return "\n".join(text_blocks)
    else:
        raise ValueError(f"parse_file: Unsupported file type => {ext}")


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
