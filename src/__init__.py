from dotenv import load_dotenv

load_dotenv("src/.env")

from concurrent.futures import ThreadPoolExecutor
executor = ThreadPoolExecutor()

from src.config import Config
config = Config()

# 延迟加载 KnowledgeBase，避免在导入时要求 API Key
_knowledge_base = None

def get_knowledge_base():
    """获取 KnowledgeBase 单例（延迟加载）"""
    global _knowledge_base
    if _knowledge_base is None:
        from src.stores import KnowledgeBase
        _knowledge_base = KnowledgeBase()
    return _knowledge_base

# 保持向后兼容的别名
knowledge_base = property(lambda self: get_knowledge_base())

def get_retriever():
    from src.knowledge.core.retriever import Retriever
    return Retriever()
