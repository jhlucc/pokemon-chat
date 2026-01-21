"""
LangChain Tool Definitions

使用 @tool 装饰器定义工具，支持 Pydantic 输入验证。
"""
from typing import Optional, Literal
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langgraph.types import Command

from src.agents.tools.websearch.websearcher import LiteBaseSearcher
from src.models.schemas import Source

# ============================================================
# Tool Input Schemas
# ============================================================

class WebSearchInput(BaseModel):
    """Web 搜索工具输入"""
    query: str = Field(description="搜索关键词")
    top_k: int = Field(default=3, description="返回结果数量")

class WeatherInput(BaseModel):
    """天气查询工具输入"""
    location: str = Field(description="城市名称")
    units: Literal["celsius", "fahrenheit"] = Field(default="celsius", description="温度单位")

# ============================================================
# Tools
# ============================================================

@tool(args_schema=WebSearchInput)
def web_search(query: str, top_k: int = 3) -> str:
    """
    联网搜索相关信息。
    
    用于查询最新资讯、实时信息、或知识库中没有的内容。
    """
    searcher = LiteBaseSearcher()
    results = searcher.search(query, top_k=top_k)
    
    if not results:
        return f"未找到关于 '{query}' 的相关信息。"
    
    # Format results
    formatted = "\n".join([
        f"- [{r.title}]({r.url or '#'}): {r.content_snippet or ''}"
        for r in results
    ])
    return f"搜索结果:\n{formatted}"

@tool
def clear_conversation_history() -> Command:
    """
    清除当前对话历史。
    
    这是一个敏感操作，通常需要用户确认。
    """
    # 返回 Command 来更新状态
    return Command(update={"messages": []})

@tool
def get_current_time() -> str:
    """
    获取当前时间。
    """
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# ============================================================
# Tool Registry
# ============================================================

ALL_TOOLS = [
    web_search,
    clear_conversation_history,
    get_current_time,
]

def get_tool_by_name(name: str):
    """根据名称获取工具"""
    for t in ALL_TOOLS:
        if t.name == name:
            return t
    return None
