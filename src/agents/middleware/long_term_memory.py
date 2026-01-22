from typing import List, Dict, Any
from langchain_core.messages import SystemMessage, BaseMessage, HumanMessage, AIMessage

from src.agents.middleware.base import BaseMiddleware, MiddlewareContext
from src.memory.semantic_memory import SemanticMemoryManager
from src.utils.logger import get_logger

logger = get_logger(__name__)

class LongTermMemoryMiddleware(BaseMiddleware):
    def __init__(self, memory_manager: SemanticMemoryManager = None):
        self.memory_manager = memory_manager or SemanticMemoryManager()
        
    def before_model(self, messages: list, context: MiddlewareContext) -> list:
        """模型调用前：检索相关长期记忆并注入"""
        # 找到最后一条用户消息
        last_human_msg = next((m for m in reversed(messages) if isinstance(m, HumanMessage)), None)
        if not last_human_msg:
            return messages
            
        query = last_human_msg.content
        
        # 检索记忆
        memories = self.memory_manager.search_memory_sync(query, k=3)
        
        if memories:
            memory_context = "\n".join([f"- {doc.page_content}" for doc in memories])
            system_injection = f"\n\n【长期记忆】\n用户过往提到的偏好与信息：\n{memory_context}\n请在回答时适当参考这些信息，表现出你记得用户。"
            
            # 注入到 SystemMsg
            if isinstance(messages[0], SystemMessage):
                messages[0] = SystemMessage(content=messages[0].content + system_injection)
            else:
                messages.insert(0, SystemMessage(content=system_injection))
                
            logger.info(f"注入了 {len(memories)} 条长期记忆")
            
        return messages

    def after_model(self, response: Any, context: MiddlewareContext) -> Any:
        """模型调用后：保存对话到长期记忆"""
        # 注意: 这里无法直接获得 user query，需要从 context 或 outside state 获得
        # 但 MiddlewareContext 目前比较简单。
        # 我们可以暂且跳过这里，使用 after_agent 钩子? 
        # BaseMiddleware definition actually allows access to state in after_agent
        return response

    def after_agent(self, state: Dict[str, Any], context: MiddlewareContext) -> Dict[str, Any]:
        """Agent执行后：保存本轮对话"""
        messages = state.get("messages", [])
        if len(messages) < 2:
            return state
            
        last_msg = messages[-1]
        second_last = messages[-2]
        
        if isinstance(last_msg, AIMessage) and isinstance(second_last, HumanMessage):
             # 简单的保存策略：保存 Q+A
             # 实际生产中应使用 LLM 总结 nugget
             memory_text = f"User: {second_last.content}\nAI: {last_msg.content}"
             self.memory_manager.add_memory_sync(memory_text, metadata={
                 "user_id": context.user_id,
                 "thread_id": context.thread_id,
                 "source": context.agent_name
             })
             logger.debug("已保存长期记忆")
             
        return state
