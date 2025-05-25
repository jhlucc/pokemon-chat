from .history_chat import *
from .indexing import *
from .Milvus import *
from rag.core.vectorrecall import VectorRecaller  # 知识库
def get_kg_agent():
    from agent.kg_agent import KGQueryAgent
    return KGQueryAgent()
