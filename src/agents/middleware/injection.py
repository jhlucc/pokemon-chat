from typing import List, Callable, Any, Dict
from langchain_core.messages import BaseMessage, SystemMessage

from src.agents.middleware.base import BaseMiddleware, MiddlewareContext
from src.utils.logger import LogManager

logger = LogManager()

class InjectionMiddleware(BaseMiddleware):
    """
    上下文注入中间件
    
    允许在模型调用前注入额外的上下文信息 (如 SystemPrompt, RAG 结果等)。
    """
    
    def __init__(self, injectors: List[Callable[[MiddlewareContext], BaseMessage]] = None):
        """
        :param injectors: 注入器函数列表。每个函数接收 context 并返回一个 Message (通常是 SystemMessage)。
                          如果返回 None，则不注入。
        """
        super().__init__()
        self.injectors = injectors or []
        
    def add_injector(self, injector: Callable[[MiddlewareContext], BaseMessage]):
        self.injectors.append(injector)

    def before_model(self, messages: List[BaseMessage], context: MiddlewareContext) -> List[BaseMessage]:
        """
        在调用模型前执行注入逻辑
        """
        injected_messages = []
        
        for injector in self.injectors:
            try:
                msg = injector(context)
                if msg and isinstance(msg, BaseMessage):
                    injected_messages.append(msg)
            except Exception as e:
                logger.error(f"Context injection failed: {e}")
                
        # 通常我们将注入的信息放在最前面 (SystemPrompt) 或者 UserMessage 之前
        # 这里简单策略：如果是 SystemMessage 放在最前，其他放在最后(但在 User 之前? 
        # LangChain模型通常接受 SystemMessage 在首位)
        
        final_messages = []
        
        # 分离出原有的 SystemMessages
        system_msgs = [m for m in messages if isinstance(m, SystemMessage)]
        other_msgs = [m for m in messages if not isinstance(m, SystemMessage)]
        
        new_system_msgs = [m for m in injected_messages if isinstance(m, SystemMessage)]
        other_injected = [m for m in injected_messages if not isinstance(m, SystemMessage)]
        
        # 组装: [Original System] + [New System] + [Other Injected] + [Original Others]
        # 或者: [New System] + [Original System] ... 取决于优先级
        # 这里假设注入的 System 提示优先级较高，或者作为补充
        
        final_messages.extend(system_msgs)
        final_messages.extend(new_system_msgs)
        # 将非 System 的注入消息放在历史记录之前 (例如检索到的相关文档)
        final_messages.extend(other_injected)
        final_messages.extend(other_msgs)
        
        return final_messages
