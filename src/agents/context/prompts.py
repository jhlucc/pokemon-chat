from collections.abc import Callable
from functools import wraps
from typing import Any

from pydantic import BaseModel


class PromptConfig(BaseModel):
    """Prompt 配置类"""

    template: str
    required_keys: list[str] = []

    def format(self, **kwargs) -> str:
        """格式化 prompt，检查必要参数"""
        missing = [key for key in self.required_keys if key not in kwargs]
        if missing:
            raise ValueError(f"Missing required prompt keys: {missing}")
        return self.template.format(**kwargs)


class DynamicPromptRegistry:
    """动态 Prompt 注册表"""

    _registry: dict[str, Callable] = {}

    @classmethod
    def register(cls, name: str):
        def decorator(func):
            cls._registry[name] = func

            @wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)

            return wrapper

        return decorator

    @classmethod
    def get(cls, name: str) -> Callable | None:
        return cls._registry.get(name)


def dynamic_prompt(name: str):
    """装饰器：标记函数为动态 Prompt 生成器"""
    return DynamicPromptRegistry.register(name)


# 示例: 状态感知 Prompt
@dynamic_prompt("state_aware")
def state_aware_prompt(state: dict[str, Any], base_prompt: str) -> str:
    """
    根据对话状态动态调整 System Prompt
    例如: 如果对话过长，提示 Agent 总结
    """
    messages = state.get("messages", [])
    if len(messages) > 20:
        return f"{base_prompt}\n\n[注意] 对话历史较长，请尽量简洁回答或总结重点。"
    return base_prompt
