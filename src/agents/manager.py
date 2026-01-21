# src/agents/manager.py

from src.agents.chat_agent import PokemonKGChatAgent
# from src.agents.react_agent import ReActAgent

class AgentManager:
    def __init__(self):
        self.agents = {}
        self.instances = {}


    def register(self, name: str, agent_class):
        self.agents[name] = agent_class

    def get_agent(self, name: str, force_new: bool = False):
        if name not in self.agents:
            raise ValueError(f"Agent '{name}' 未注册")
        
        if force_new:
            return self.agents[name]()
            
        if name not in self.instances:
            self.instances[name] = self.agents[name]()
            
        return self.instances[name]

    def list_agents(self):
        return [self.get_agent(name).get_info() for name in self.agents]

# 初始化并注册
agent_manager = AgentManager()
agent_manager.register("chat_agent", PokemonKGChatAgent)
# agent_manager.register("react_agent", ReActAgent)  # 如果不需要可以注释
