import os

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.knowledge.ingestion.parsers.base import parse_file
from src.knowledge.store.vector import VectorStore
from src.utils.logger import get_logger

_log = get_logger(__name__)

# Pokemon-specific separators for semantic-aware chunking
# Priority: double newlines (paragraphs) > headers > sentences > words
_POKEMON_SEPARATORS = [
    "\n\n",  # Paragraph breaks
    "\n## ",  # Markdown H2 headers
    "\n### ",  # Markdown H3 headers
    "\n",  # Line breaks
    "。",  # Chinese period
    "！",  # Chinese exclamation
    "？",  # Chinese question mark
    ". ",  # English period
    "! ",  # English exclamation
    "? ",  # English question mark
    "；",  # Chinese semicolon
    "; ",  # English semicolon
    "，",  # Chinese comma
    ", ",  # English comma
    " ",  # Spaces
    "",  # Character-level fallback
]


def _get_splitter(chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    """Get a semantic-aware text splitter with Pokemon-specific separators."""
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=_POKEMON_SEPARATORS,
        length_function=len,
        is_separator_regex=False,
    )


class IngestionPipeline:
    def __init__(self, vector_store: VectorStore):
        self.store = vector_store

    def ingest_file(
        self,
        file_path: str,
        chunk_size: int = 800,  # Smaller chunks for better retrieval matching
        chunk_overlap: int = 150,  # More overlap to preserve context
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
