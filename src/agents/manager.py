from typing import Dict, Type, Any, Optional
from src.agents.base import BaseAgent
from src.utils.logger import get_logger

logger = get_logger(__name__)


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
    
    所有注册的 Agent 类必须继承自 BaseAgent。
    """
    
    def __init__(self):
        self._registry: Dict[str, Type[BaseAgent]] = {}
        self._instances: Dict[str, BaseAgent] = {}
        
        # 延迟导入以避免循环依赖
        self._register_builtin_agents()

    def _register_builtin_agents(self):
        """注册内置 Pokemon 主题 Agent"""
        try:
            from src.agents.chat_agent import PokemonKGChatAgent
            self.register("chat_agent", PokemonKGChatAgent)
        except Exception as e:
            logger.warning(f"Failed to register chat_agent: {e}")
        
        try:
            from src.agents.deep_agent import DeepAgent
            self.register("deep_agent", DeepAgent)
        except Exception as e:
            logger.warning(f"Failed to register deep_agent: {e}")
        
        try:
            from src.agents.pokemon_stats_agent import PokemonStatsAgent
            self.register("stats_agent", PokemonStatsAgent)
        except Exception as e:
            logger.warning(f"Failed to register stats_agent: {e}")
        
        try:
            from src.agents.pokedex_agent import PokedexAgent
            self.register("pokedex_agent", PokedexAgent)
        except Exception as e:
            logger.warning(f"Failed to register pokedex_agent: {e}")
        
        try:
            from src.agents.trainer_agent import TrainerAgent
            self.register("trainer_agent", TrainerAgent)
        except Exception as e:
            logger.warning(f"Failed to register trainer_agent: {e}")

    def register(self, name: str, agent_cls: Type[BaseAgent]):
        """注册 Agent 类"""
        self._registry[name] = agent_cls
        logger.info(f"Registered agent: {name} -> {agent_cls.__name__}")

    def unregister(self, name: str) -> bool:
        """注销 Agent 类"""
        if name in self._registry:
            del self._registry[name]
            if name in self._instances:
                del self._instances[name]
            logger.info(f"Unregistered agent: {name}")
            return True
        return False

    def get_agent(self, name: str, **kwargs) -> BaseAgent:
        """
        获取 Agent 实例 (懒加载)
        
        Args:
            name: Agent 名称
            **kwargs: 传递给 Agent 构造函数的额外参数
        """
        # 如果已有实例且无额外参数，直接返回
        if name in self._instances and not kwargs:
            return self._instances[name]
        
        if name not in self._registry:
            raise ValueError(f"Agent '{name}' not found in registry. Available: {list(self._registry.keys())}")
            
        # Instantiate
        try:
            agent_cls = self._registry[name]
            instance = agent_cls(**kwargs) if kwargs else agent_cls()
            
            # 只缓存无参数创建的实例
            if not kwargs:
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
                agent = self.get_agent(name)
                info_list[name] = agent.get_info()
            except Exception as e:
                info_list[name] = {"error": str(e)}
        return info_list

    def list_registered(self) -> list:
        """列出所有已注册的 Agent 名称"""
        return list(self._registry.keys())

    def is_registered(self, name: str) -> bool:
        """检查 Agent 是否已注册"""
        return name in self._registry


# Export singleton instance
agent_manager = AgentManager()

