from typing import List, Dict, Any
from langchain_core.messages import SystemMessage, BaseMessage, HumanMessage, AIMessage

from src.agents.middleware.base import BaseMiddleware, MiddlewareContext
from src.knowledge.memory.semantic import SemanticMemoryManager
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

    async def after_agent(self, state: Dict[str, Any], context: MiddlewareContext) -> Dict[str, Any]:
        """Agent执行后：保存本轮对话 (使用后台线程总结)"""
        messages = state.get("messages", [])
        if len(messages) < 2:
            return state
            
        # 启动后台线程进行总结和保存
        import threading
        thread = threading.Thread(
            target=self._background_save_memory,
            args=(messages, context)
        )
        thread.daemon = True
        thread.start()
              
        return state

    def _background_save_memory(self, messages: list, context: MiddlewareContext):
        """后台处理：总结并保存记忆"""
        try:
            # 1. 提取最近对话
            # 简单的策略：取最后2轮
            recent_msgs = messages[-4:] if len(messages) >= 4 else messages[-2:]
            
            # 格式化对话文本
            conversation_text = ""
            for msg in recent_msgs:
                role = "User" if isinstance(msg, HumanMessage) else "AI"
                content = msg.content
                conversation_text += f"{role}: {content}\n"

            # 2. 使用 LLM 总结 (需要 access to LLM)
            # 这里我们需要一个 LLM 实例。简单起见，如果 middleware 初始化时没有 LLM，
            # 我们可以尝试从 context 获取，或者单独初始化一个轻量级 LLM。
            # 为演示，这里假设 memory_manager 有某种 smart add 或者是直接保存 raw for now 
            # (优化计划中提到要 summarize, 所以我们尝试实例化一个 LLM)

            from langchain_openai import ChatOpenAI
            from src.core.settings import settings
            
            # 使用轻量模型进行总结
            llm = ChatOpenAI(
                model=settings.llm.model_name, 
                base_url=settings.llm.api_base,
                api_key=settings.llm.api_key,
                temperature=0.1
            )
            
            prompt = f"""
            请从以下对话中提取用户的重要偏好、事实或正在进行的任务状态。
            忽略客套话。如果无重要信息，返回 "无"。
            
            对话内容:
            {conversation_text}
            
            总结 (不超过50字):
            """
            
            response = llm.invoke(prompt)
            summary = response.content.strip()
            
            if summary and summary != "无":
                logger.info(f"[LongTermMemory] 生成记忆摘要: {summary}")
                 # 3. 保存到向量数据库
                self.memory_manager.add_memory_sync(summary, metadata={
                    "user_id": context.user_id,
                    "thread_id": context.thread_id,
                    "source": context.agent_name + "_summary",
                    "timestamp": str(context.thread_id) # pseudo timestamp substitute
                })
            else:
                 logger.debug("[LongTermMemory] 无重要记忆需保存")

        except Exception as e:
            logger.error(f"[LongTermMemory] 后台保存失败: {e}")
