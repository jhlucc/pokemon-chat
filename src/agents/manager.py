from typing import Dict, Type, Any, Optional
from src.agents.base import BaseAgent
from src.utils.logger import LogManager
from src.agents.chat_agent import PokemonKGChatAgent
from src.agents.deep_agent import DeepAgent

logger = LogManager()

class SingletonMeta(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

class AgentManager(metaclass=SingletonMeta):
    """
    Agent 管理器 (单例)
    负责注册、获取和管理 Agent 实例。
    """
    
    def __init__(self):
        self._registry: Dict[str, Type[BaseAgent]] = {}
        self._instances: Dict[str, BaseAgent] = {}
        
        # 内置注册
        # 注意: 这里使用字符串作为 key
        self.register("chat_agent", PokemonKGChatAgent)
        self.register("deep_agent", DeepAgent)

    def register(self, name: str, agent_cls: Type[BaseAgent]):
        """注册 Agent 类"""
        self._registry[name] = agent_cls
        logger.info(f"Registered agent: {name} -> {agent_cls.__name__}")

    def get_agent(self, name: str) -> BaseAgent:
        """获取 Agent 实例 (懒加载)"""
        if name in self._instances:
            return self._instances[name]
        
        if name not in self._registry:
            raise ValueError(f"Agent '{name}' not found in registry. Available: {list(self._registry.keys())}")
            
        # Instantiate
        try:
            agent_cls = self._registry[name]
            # 这里假定无参构造，或者将来支持参数传递
            instance = agent_cls() 
            self._instances[name] = instance
            logger.info(f"Instantiated agent: {name}")
            return instance
        except Exception as e:
            logger.error(f"Failed to instantiate agent '{name}': {e}")
            raise e

    def list_agents(self) -> Dict[str, Any]:
        """列出所有已注册 Agent 的信息"""
        info_list = {}
        for name in self._registry:
            try:
                # 获取实例以读取 info
                agent = self.get_agent(name)
                info_list[name] = agent.get_info()
            except Exception as e:
                info_list[name] = {"error": str(e)}
        return info_list

# Export singleton instance
agent_manager = AgentManager()
