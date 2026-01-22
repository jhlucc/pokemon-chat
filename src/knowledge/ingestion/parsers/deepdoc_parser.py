from src.plugins.parser import PdfParser, DocxParser, ExcelParser, PptParser
from src.utils.logger import get_logger

_log = get_logger(__name__)

class DeepDocParser:
    @staticmethod
    def parse(file_path: str) -> str:
        ext = file_path.split('.')[-1].lower()
        if not ext.startswith('.'):
             ext = '.' + ext
             
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
                
        return "\n".join(all_text)
