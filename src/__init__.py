from dotenv import load_dotenv

load_dotenv("src/.env")

from concurrent.futures import ThreadPoolExecutor
executor = ThreadPoolExecutor()

from src.config import Config
config = Config()

from src.stores import KnowledgeBase
knowledge_base = KnowledgeBase()

def get_retriever():
    from rag.core.retriever import Retriever
    return Retriever()
