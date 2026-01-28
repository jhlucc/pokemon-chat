import os

from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter

from src.knowledge.ingestion.parsers.base import parse_file
from src.knowledge.store.vector import VectorStore
from src.utils.logger import get_logger

_log = get_logger(__name__)
_warned_tiktoken = False


def _get_splitter(chunk_size: int, chunk_overlap: int) -> CharacterTextSplitter:
    global _warned_tiktoken
    try:
        return CharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    except Exception as exc:
        if not _warned_tiktoken:
            _log.warning("tiktoken splitter unavailable; falling back to char splitter. (%s)", exc)
            _warned_tiktoken = True
        return CharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )


class IngestionPipeline:
    def __init__(self, vector_store: VectorStore):
        self.store = vector_store

    def ingest_file(
        self,
        file_path: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
        do_ocr: bool = False,
        file_id: str | None = None,
    ) -> str:
        # 1. Parse
        _log.info(f"Pipeline: Parsing {file_path}")
        text = parse_file(file_path, do_ocr=do_ocr)

        # 2. Chunk
        _log.info(f"Pipeline: Chunking text (len={len(text)})")
        splitter = _get_splitter(chunk_size, chunk_overlap)
        chunks = splitter.split_text(text)

        # 3. Create Documents
        docs = []
        fid = file_id or os.path.basename(file_path)

        for idx, chunk in enumerate(chunks):
            metadata = {"source": file_path, "file_id": fid, "chunk_index": idx}
            docs.append(Document(page_content=chunk, metadata=metadata))

        # 4. Insert into Store
        if docs:
            _log.info(f"Pipeline: Inserting {len(docs)} chunks into store.")
            self.store.insert(docs)

        return fid

    def ingest_directory(self, dir_path: str, suffix_list: list[str] = None):
        if not suffix_list:
            suffix_list = [".pdf", ".docx", ".txt", ".md", ".ppt", ".pptx"]

        for root, _, files in os.walk(dir_path):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in suffix_list:
                    path = os.path.join(root, file)
                    try:
                        self.ingest_file(path)
                    except Exception as e:
                        _log.error(f"Failed to ingest {path}: {e}")
