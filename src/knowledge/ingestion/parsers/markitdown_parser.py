from markitdown import MarkItDown

class MarkItDownParser:
    @staticmethod
    def parse(file_path: str) -> str:
        md = MarkItDown()
        result = md.convert(file_path)
        if result and result.text_content:
            return result.text_content
        return ""
