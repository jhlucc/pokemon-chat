
class BaseAgent:
    """
    所有 Agent 都应继承这个基类，并实现以下方法：
    - get_info(): 返回给前端的元数据（名称、描述、工具等）
    - query(): 核心的问答调用接口
    """

    def get_info(self) -> dict:
        raise NotImplementedError("Agent 必须实现 get_info() 方法")

    async def query(self, question: str, **kwargs):
        raise NotImplementedError("Agent 必须实现 query() 方法")
