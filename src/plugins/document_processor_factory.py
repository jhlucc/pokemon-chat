from src.plugins.document_processor_base import BaseDocumentProcessor
from src.utils.logger import LogManager

logger = LogManager()


class DocumentProcessorFactory:
    """文档处理器工厂"""

    _registry: dict[str, type[BaseDocumentProcessor]] = {}

    @classmethod
    def register(cls, name: str, processor_cls: type[BaseDocumentProcessor]):
        """注册新的处理器"""
        cls._registry[name] = processor_cls
        logger.info(f"Registered document processor: {name}")

    @classmethod
    def get_processor(cls, name: str) -> BaseDocumentProcessor:
        """获取处理器实例"""
        if name not in cls._registry:
            raise ValueError(f"Unknown processor type: {name}. Available: {list(cls._registry.keys())}")

        processor_cls = cls._registry[name]
        return processor_cls()
