import asyncio
import sys
from pathlib import Path
from typing import Dict, Optional, List, Any

# LangChain Imports
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI

# LangGraph Imports
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, MessagesState, START, END

# Project Imports
from src.core.settings import settings
from src.agents.base import BaseAgent
from src.knowledge import PokemonLightRAG, get_lightrag_instance
from src.agents.tools.websearch.websearcher import LiteBaseSearcher
from src.agents.kg_agent import KGQueryAgent
from src.utils.logger import LogManager
from src.models.schemas import AgentResponse
from src.agents.interrupts import approval_node



# Middleware Imports
from src.agents.middleware import (
    LoggingMiddleware, 
    RetryMiddleware, 
    FallbackMiddleware, 
    MemoryMiddleware, 
    MiddlewareChain, 
    MiddlewareContext
)

logger = LogManager()

# 状态定义
class AgentState(MessagesState):
    next: str
    user_id: Optional[str]
    thread_id: Optional[str]
    response_mode: Optional[str] # "text", "json", "markdown"
    approval_status: Optional[str] # "approved", "rejected"
    user_feedback: Optional[str]


class PokemonKGChatAgent(BaseAgent):
    """宝可梦知识图谱聊天代理"""

    def __init__(
        self, 
        openai_base_url: str = None,
        openai_api_key: str = None,
        model_name: str = None
    ):
        # 优先使用传入参数，否则使用全局配置
        self.openai_base_url = openai_base_url or settings.llm.api_base
        self.openai_api_key = openai_api_key or settings.llm.api_key
        self.model_name = model_name or settings.llm.model_name
        
        # 初始化中间件
        self._init_middleware()
        
        # 初始化组件
        self._init_components()
        
        # 初始化 Checkpointer
        self._init_checkpointer()
        
        # 构建图
        self._build_graph()

    def _init_checkpointer(self):
        """初始化 Checkpointer"""
        type_ = settings.agent.checkpointer_type.lower()
        if type_ == "memory":
            self.checkpointer = MemorySaver()
        elif type_ == "sqlite":
            try:
                from langgraph.checkpoint.sqlite import SqliteSaver
                import sqlite3
                conn = sqlite3.connect(":memory:", check_same_thread=False) 
                # 注意: 真正的 SQLite 持久化需要文件路径，这里暂时演示用 :memory: 或者需要配置路径
                # 由于环境限制，暂时使用 MemorySaver 替代不支持的情况
                self.checkpointer = SqliteSaver(conn)
            except ImportError:
                logger.warning("langgraph-checkpoint-sqlite not found, falling back to MemorySaver")
                self.checkpointer = MemorySaver()
        else:
            logger.warning(f"Unknown checkpointer type '{type_}', falling back to MemorySaver")
            self.checkpointer = MemorySaver()

    def _init_middleware(self):
        """初始化中间件链"""
        self.middleware = MiddlewareChain()
        
        # 1. 日志中间件
        self.middleware.add(LoggingMiddleware(log_messages=True))
        
        # 2. 记忆管理中间件
        self.middleware.add(MemoryMiddleware(
            max_messages=settings.agent.conversation_max_messages,
            strategy="trim"
        ))
        
        # 3. 重试中间件
        self.middleware.add(RetryMiddleware(max_retries=2))

    def _init_components(self):
        """初始化所有组件"""
        # 初始化 LLM - 包装重试和回退中间件
        self.base_llm = ChatOpenAI(
            model=self.model_name,
            base_url=self.openai_base_url,
            api_key=self.openai_api_key,
            temperature=settings.llm.temperature,
            max_tokens=settings.llm.max_tokens
        )
        
        # 使用中间件包装 LLM 调用
        context = MiddlewareContext(agent_name="chat_agent")
        # 包装 Invoke 方法以支持重试等中间件逻辑
        # 使用 RunnableLambda 保持 LangChain 兼容性
        wrapped_invoke = self.middleware.wrap_model_call(self.base_llm.invoke, context)
        self.llm = RunnableLambda(wrapped_invoke)


        # 初始化知识图谱查询代理
        self.kgsql_agent = None
        if settings.features.enable_knowledge_graph:
            try:
                self.kgsql_agent = KGQueryAgent(llm=self.llm)
            except Exception as e:
                logger.error(f"Failed to initialize KGQueryAgent: {e}")


        # 初始化图RAG (LightRAG)
        self.lightrag = get_lightrag_instance(
            workspace="pokemon_kb",
            working_dir=str(settings.paths.artifacts_data),
        )

        # 初始化搜索工具
        self.searcher = LiteBaseSearcher()

        # 添加兼容方法 (模拟搜索)
        async def fake_search_and_generate(query: str) -> str:
            if settings.features.enable_web_search:
                # 如果开启了 web search，尝试真实的搜索
                try:
                    # LiteBaseSearcher.search 是同步方法，但可以在 async 中直接调用
                    results = self.searcher.search(query, top_k=3)
                    if not results:
                        return f"未能搜索到关于 '{query}' 的相关信息。"
                    
                    # 格式化结果
                    
                    # ])
                    # return f"关于 '{query}' 的搜索结果：\n{formatted}"
                    
                    # 兼容 Source 对象列表
                    from src.models.schemas import Source
                    if results and isinstance(results[0], Source):
                         formatted = "\n".join([
                            f"- [{r.title}]({r.url or '#'}): {r.content_snippet}"
                            for r in results
                        ])
                    else:
                        formatted = "\n".join([
                            f"- [{r.get('title', '未知标题')}]({r.get('url', '#')}): {r.get('content', '')}"
                            for r in results
                        ])
                    return f"关于 '{query}' 的搜索结果：\n{formatted}"
                except Exception as e:
                    logger.error(f"Web search failed: {e}")
                    return f"搜索失败: {e}"
            return f"（模拟联网搜索结果，无实际联网）Query: {query}"

        # 检查 searcher 是否有 search_and_generate
        if not hasattr(self.searcher, "search_and_generate"):
             self.searcher.search_and_generate = fake_search_and_generate

    def _build_graph(self):
        """构建LangGraph状态图"""
        builder = StateGraph(AgentState)

        # 添加节点
        builder.add_node("supervisor", self._supervisor)
        builder.add_node("chat", self._chat)
        builder.add_node("kg_sqler", self._kgsql_node)
        builder.add_node("graph_rager", self._graph_rager)
        builder.add_node("web_searcher", RunnableLambda(self._web_searcher))
        builder.add_node("approval", approval_node)

        # 定义成员列表
        self.members = ["chat", "graph_rager", "web_searcher"]
        if self.kgsql_agent:
            self.members.append("kg_sqler")

        # 添加边
        for member in self.members:
            builder.add_edge(member, "supervisor")

        # 添加条件边
        def route_supervisor(state):
            next_node = state["next"]
            # 示例: 如果 supervisor 指定 approval，或者我们在某处检测到敏感操作
            if next_node == "approval":
                return "approval"
            return next_node

        builder.add_conditional_edges("supervisor", route_supervisor)
        # approval 之后通常回到 supervisor 或继续执行
        builder.add_edge("approval", "supervisor")
        
        builder.add_edge(START, "supervisor")

        # 编译图 - 使用 checkpointer
        self._graph = builder.compile(checkpointer=self.checkpointer)

    @property
    def graph(self):
        return self._graph

    # 节点函数定义 ... (后续补全)
    
    def _chat(self, state: AgentState):
        """自然语言聊天节点"""
        messages = state["messages"]
        # 应用中间件
        context = MiddlewareContext(
            agent_name="chat_agent",
            thread_id=state.get("thread_id", ""),
            user_id=state.get("user_id", "")
        )
        messages = self.middleware.run_before_model(messages, context)
        
        # 获取响应模式，默认为 text
        response_mode = state.get("response_mode", "text")
        
        try:
            if response_mode == "json":
                # 结构化输出模式
                # 1. 创建结构化 LLM
                structured_base_llm = self.base_llm.with_structured_output(AgentResponse)
                
                # 2. 也是用 middleware 包装它，保证 retry/logging 生效
                # 注意：wrap_model_call 包装的是一个 callable (input -> output)
                wrapped_structured_invoke = self.middleware.wrap_model_call(structured_base_llm.invoke, context)
                
                # 3. 调用
                model_response = wrapped_structured_invoke(messages)
                
                # 如果成功返回对象，转换为 JSON 字符串放入 content
                if isinstance(model_response, AgentResponse):
                    content = model_response.model_dump_json()
                else:
                    # 某些情况下可能直接返回了 dict 或其他
                    import json
                    content = json.dumps(model_response, ensure_ascii=False) if isinstance(model_response, dict) else str(model_response)
                
                model_response = AIMessage(content=content)
            else:
                # 默认文本模式
                model_response = self.llm.invoke(messages)

        except Exception as e:
            logger.error(f"LLM invoke failed (mode={response_mode}): {e}")
            if response_mode == "json":
                # JSON 模式下发生错误，返回标准错误结构
                from src.models.schemas import ErrorResponse
                error_resp = ErrorResponse(
                    error_code="llm_error", 
                    message=f"Failed to generate structured response: {str(e)}"
                )
                model_response = AIMessage(content=error_resp.model_dump_json())
            else:
                # 文本模式，直接返回错误信息
                model_response = AIMessage(content=f"Error generating response: {str(e)}")

        model_response = self.middleware.run_after_model(model_response, context)
        
        return {"messages": [model_response]}

    def _kgsql_node(self, state: AgentState):
        """知识图谱查询节点"""
        if not self.kgsql_agent:
            return {"messages": [HumanMessage(content="知识图谱功能未开启或初始化失败。", name="kg_sqler")]}

        # 注意: kgsql_agent.agent 是一个 Runnable/Agent, 也可以应用 middleware
        try:
            result = self.kgsql_agent.agent.invoke(state)
            # 提取最后一条消息
            last_msg = result["messages"][-1]
            content = last_msg.content if hasattr(last_msg, "content") else str(last_msg)
            
            return {
                "messages": [
                    HumanMessage(content=content, name="kg_sqler")
                ]
            }
        except Exception as e:
            logger.error(f"KG SQL Error: {e}")
            return {"messages": [HumanMessage(content=f"查询知识图谱失败: {e}", name="kg_sqler")]}

    async def _graph_rager(self, state: AgentState):
        """图RAG查询节点 (LightRAG)"""
        messages = state["messages"]
        last_content = messages[-1].content if messages else ""
        
        try:
            # Use LightRAG async query
            response = await self.lightrag.query(
                query_text=last_content,
                mode="mix",
                only_need_context=True,
                top_k=10
            )
            content = response if isinstance(response, str) else str(response)
            return {"messages": [HumanMessage(content=content, name="graph_rager")]}
        except Exception as e:
            logger.error(f"Graph RAG Error: {e}")
            return {"messages": [HumanMessage(content=f"Graph RAG 查询失败: {e}", name="graph_rager")]}

    async def _web_searcher(self, state: AgentState):
        """网络搜索节点"""
        logger.info("📡 已调用 web_searcher 节点")
        messages = state["messages"]
        last_content = messages[-1].content if messages else ""
        
        try:
            response = await self.searcher.search_and_generate(last_content)
            return {"messages": [HumanMessage(content=response, name="web_searcher")]}
        except Exception as e:
             return {"messages": [HumanMessage(content=f"网络搜索失败: {e}", name="web_searcher")]}

    def _supervisor(self, state: AgentState):
        """监督员节点"""
        # (保持原有的 prompt 逻辑，但可以使用 prompt template 功能)
        system_prompt = (
            "你被指定为对话监督员，负责协调以下工作模块的协作：{members}\n\n"
            "各模块职能划分：\n"
            "- chat：自然语言交互模块\n"
            "  • 直接处理用户输入的自然语言响应\n"
            "- kg_sqler：宝可梦知识图谱查询模块\n"
            "  • 属性数据（种族值/进化链/特性）\n"
            "  • 角色关系（训练师/劲敌/团队）\n"
            "  • 地域情报（地点/道馆/栖息地）\n"
            "- graph_rager：宝可梦知识库(RAG)\n"
            "  • 人物介绍、社群发现、路径分析、时序关联\n"
            "- web_searcher：实时联网搜索模块\n"
            "  • 最新资讯、新闻、时效性内容、社区讨论\n\n"
            "模块调用原则：\n"
            "1. 优先使用本地知识库(kg_sqler/graph_rager)\n"
            "2. 涉及最新/外部信息时调用 web_searcher\n"
            "3. 无法回答时调用 chat 进行一般对话\n"
            "4. 每个模块执行后将返回任务结果及状态。\n"
            "执行流程规范：\n"
            "1. chat模块最多能调用一次\n"
            "2. 可以链式调用多个模块\n"
            "3. 当某个模块结果不足时，继续调用其他模块\n"
            "4. 任务完成时，返回 FINISH 终止符\n"
            "5. 当用户询问'删除记忆'或'清空历史'等敏感操作时，返回 'approval' 模块进行确认\n"
        )

        prompt = ChatPromptTemplate.from_template("""
        {system_prompt}
        
        请严格按以下JSON格式回复，只包含next字段:
        {{
            "next": "FINISH"
        }}
        或者
        {{
            "next": "模块名称"
        }}
        
        输入：{input}
        """)

        # 获取最后几条消息作为输入 context
        messages = state["messages"]
        # 应用中间件修剪 (虽然 _init_middleware 已经加了, 但这里可以再次确保)
        
        chain = prompt | self.llm | JsonOutputParser()
        
        # 动态构建 members 描述
        members_desc = ", ".join(self.members)
        
        try:
            response = chain.invoke({
                "system_prompt": system_prompt.format(members=members_desc),
                "input": messages
            })
            next_ = response.get("next", "FINISH")
        except Exception as e:
            logger.error(f"Supervisor parsing error: {e}")
            next_ = "FINISH"

        return {"next": END if next_ == "FINISH" else next_}

    # 公共接口
    async def query(
        self,
        question: str,
        meta: Optional[Dict[str, Any]] = None,
        history: Optional[List[Dict[str, Any]]] = None
    ):
        meta = meta or {}
        thread_id = meta.get("thread_id", "default_thread")
        user_id = meta.get("user_id", "user")
        response_mode = meta.get("response_mode", "text")

        input_message = {
            "messages": [HumanMessage(content=question)],
            "thread_id": thread_id,
            "user_id": user_id,
            "response_mode": response_mode,
            "next": START 
        }
        
        config = {
            "configurable": {
                "thread_id": thread_id,
                "user_id": user_id,
                # "checkpoint_ns": "" 
            }
        }

        # 运行 middleware before_agent hook
        context = MiddlewareContext(
            agent_name="chat_agent", 
            thread_id="0",
            user_id="user"
        )
        input_message = self.middleware.run_before_agent(input_message, context)

        chunks = []
        try:
            async for chunk in self.graph.astream(input_message, config, stream_mode="values"):
                if "messages" in chunk and chunk["messages"]:
                    chunks.append(chunk["messages"][-1])
        except Exception as e:
            logger.error(f"Agent execution error: {e}")
            yield f"发生错误: {e}"
            return

        # 运行 middleware after_agent hook
        self.middleware.run_after_agent(input_message, context)

        yield chunks[-1].content if chunks else None

    def get_info(self):
        return {
            "name": "chat_agent",
            "description": "宝可梦图谱智能体",
            "requirements": ["NEO4J_URI"],
            "all_tools": ["graph_query", "retrieval", "web_search"]
        }

    # Time Travel APIs
    async def get_state(self, thread_id: str):
        """获取当前状态"""
        config = {"configurable": {"thread_id": thread_id}}
        return await self.graph.aget_state(config)

    async def get_state_history(self, thread_id: str, limit: int = 10):
        """获取状态历史"""
        config = {"configurable": {"thread_id": thread_id}}
        return [s async for s in self.graph.aget_state_history(config, limit=limit)]

    async def update_state(self, thread_id: str, values: dict, as_node: str = None):
        """更新状态 (时间旅行/人工干预)"""
        config = {"configurable": {"thread_id": thread_id}}
        return await self.graph.aupdate_state(config, values, as_node=as_node)

# 使用示例
if __name__ == "__main__":
    async def main():
        # 初始化代理
        agent = PokemonKGChatAgent()
        # 示例查询
        question = "拥有皮卡丘的角色中，有哪些是小刚的伙伴？"

        print(f"\n问题: {question}")
        print("回答:")
        async for chunk in agent.query(question):
            print(chunk)


    asyncio.run(main())
