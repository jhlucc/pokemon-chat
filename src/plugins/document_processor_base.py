from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel


class ProcessingResult(BaseModel):
    """文档处理结果"""

    content: str
    metadata: dict[str, Any] = {}
    error: str | None = None


class BaseDocumentProcessor(ABC):
    """
    文档处理器基类
    所有具体的文档解析器（OCR, PDFParser等）都应继承此类
    """

    @abstractmethod
    def process_file(self, file_path: str, params: dict[str, Any] = None) -> ProcessingResult:
        """
        处理单个文件
        :param file_path: 文件绝对路径
        :param params: 处理参数
        :return: 处理结果
        """
        pass

    @abstractmethod
    def check_health(self) -> dict[str, Any]:
        """
        检查处理器健康状态（例如模型是否加载，API 是否可用）
        """
        pass
