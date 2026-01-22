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
from src.knowledge import PokemonLightRAG
from src.agents.tools.websearch.websearcher import LiteBaseSearcher
from src.agents.kg_agent import KGQueryAgent
from src.agents.pokemon_stats_agent import PokemonStatsAgent
from src.agents.pokedex_agent import PokedexAgent
from src.agents.trainer_agent import TrainerAgent
from src.agents.deep_agent.graph import DeepAgent
from src.utils.logger import get_logger
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

logger = get_logger(__name__)

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
        model_name: str = None,
        **kwargs
    ):
        # 优先使用传入参数，否则使用全局配置
        self.openai_base_url = openai_base_url or settings.llm.api_base
        self.openai_api_key = openai_api_key or settings.llm.api_key
        self.model_name = model_name or settings.llm.model_name
        
        # 初始化 Checkpointer
        checkpointer = self._create_checkpointer()
        
        # 调用父类初始化 (会自动调用 _init_components 和 _build_graph)
        super().__init__(checkpointer=checkpointer, **kwargs)

    def _create_checkpointer(self):
        """初始化并返回 Checkpointer"""
        type_ = settings.agent.checkpointer_type.lower()
        if type_ == "memory":
            return MemorySaver()
        elif type_ == "sqlite":
            from langgraph.checkpoint.sqlite import SqliteSaver
            import sqlite3
            conn = sqlite3.connect(":memory:", check_same_thread=False) 
            return SqliteSaver(conn)
        else:
            raise ValueError(f"Unknown checkpointer type: {type_}")

    def _init_middleware(self):
        """初始化中间件链"""
        self.middleware = MiddlewareChain()
        
        # 日志中间件
        self.middleware.add(LoggingMiddleware(log_messages=True))
        
        # 记忆管理中间件
        self.middleware.add(MemoryMiddleware(
            max_messages=settings.agent.conversation_max_messages,
            strategy="trim"
        ))
        
        # 重试中间件
        self.middleware.add(RetryMiddleware(max_retries=2))
        
        # [NEW] 语义长期记忆中间件
        try:
            from src.agents.middleware.long_term_memory import LongTermMemoryMiddleware
            self.middleware.add(LongTermMemoryMiddleware())
            logger.info("✅ Semantic Long-Term Memory middleware added")
        except Exception as e:
            logger.error(f"❌ Failed to add LongTermMemoryMiddleware: {e}")

    def _init_components(self, **kwargs):
        """初始化所有组件"""
        # 初始化中间件
        self._init_middleware()

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
        wrapped_invoke = self.middleware.wrap_model_call(self.base_llm.invoke, context)
        self._llm = RunnableLambda(wrapped_invoke)


        # 初始化知识图谱查询代理
        self.kgsql_agent = None
        if settings.features.enable_knowledge_graph:
            try:
                # 使用 self._llm 而不是 self.llm
                self.kgsql_agent = KGQueryAgent(llm=self._llm)
            except Exception as e:
                logger.error(f"Failed to initialize KGQueryAgent: {e}")

        # [NEW] 初始化专业代理 (作为子图)
        try:
            self.stats_agent = PokemonStatsAgent()
            self.pokedex_agent = PokedexAgent()
            self.trainer_agent = TrainerAgent()
            self.deep_agent = DeepAgent()
        except Exception as e:
            logger.error(f"Failed to initialize specialized agents: {e}")

        # 初始化图RAG (LightRAG)
        self.lightrag = PokemonLightRAG(
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
        builder.add_node("guardrail", self._guardrail_node)
        builder.add_node("supervisor", self._supervisor)
        builder.add_node("chat", self._chat)
        builder.add_node("kg_sqler", self._kgsql_node)
        builder.add_node("graph_rager", self._graph_rager)
        builder.add_node("web_searcher", RunnableLambda(self._web_searcher))
        builder.add_node("approval", approval_node)

        # [NEW] 专业代理节点
        builder.add_node("stats_agent", self._stats_node)
        builder.add_node("pokedex_agent", self._pokedex_node)
        builder.add_node("trainer_agent", self._trainer_node)
        builder.add_node("deep_researcher", self._deep_research_node)

        # 定义成员列表
        self.members = ["chat", "graph_rager", "web_searcher", 
                       "stats_agent", "pokedex_agent", "trainer_agent", "deep_researcher"]
        if self.kgsql_agent:
            self.members.append("kg_sqler")

        # 添加边
        for member in self.members:
            builder.add_edge(member, "supervisor")

        # 添加条件边 for guardrail
        def route_guardrail(state):
            # guardrail node 返回的 next
            # check if it set any temp state (hacky)
            # 但 guardrail 返回了 dict with 'next' key which updates the state via "next" field (if defined in schema)
            # AgentState has 'next': str
            nxt = state.get("next")
            if nxt == "end_with_block":
                return END
            return "supervisor"

        builder.add_edge(START, "guardrail")
        builder.add_conditional_edges("guardrail", route_guardrail)

        # 添加条件边 for supervisor
        def route_supervisor(state):
            next_node = state["next"]
            if next_node == "approval":
                return "approval"
            # 兼容旧代码，如果没有 next 可能是 finish
            if not next_node:
                return END
            return next_node

        builder.add_conditional_edges("supervisor", route_supervisor)
        # approval 之后通常回到 supervisor 或继续执行
        builder.add_edge("approval", "supervisor")

        # 编译图 - 使用 checkpointer
        self._graph = builder.compile(checkpointer=self.checkpointer)
        return self._graph

    @property
    def graph(self):
        return self._graph

    # 节点函数定义 ... (后续补全)
    
    async def _guardrail_node(self, state: AgentState):
        """
        守卫节点：检查用户输入是否与 Pokemon 相关。
        """
        messages = state["messages"]
        last_user_msg = messages[-1]
        
        # 定义系统的守卫提示词
        guardrail_prompt = ChatPromptTemplate.from_template("""
        你是 Pokemon 世界的守门人。你的任务是判断用户的输入是否与 "Pokemon (宝可梦/口袋妖怪)"、"动画/游戏" 或 "日常闲聊" 相关。
        
        判断规则：
        1. 如果包含宝可梦名称、角色、招式、地点等，返回 "pass"。
        2. 如果是日常问候（你好、早上好等），返回 "pass"。
        3. 如果是完全无关的话题（如：写Python代码、政治新闻、股票分析、其他动漫等），返回 "block"。
        
        请输出 JSON 格式:
        {{
            "status": "pass" 或 "block",
            "reason": "原因"
        }}
        
        用户输入: {input}
        """)
        
        chain = guardrail_prompt | self.llm | JsonOutputParser()
        
        try:
            result = await chain.ainvoke({"input": last_user_msg.content})
            status = result.get("status", "pass")
            
            if status == "block":
                return {
                    "next": "end_with_block",
                    "messages": [AIMessage(content="抱歉，作为一个宝可梦专家，我只能回答与宝可梦相关的问题。让我们聊聊宝可梦吧！")]
                }
            return {"next": "supervisor"}
            
        except Exception as e:
            logger.error(f"Guardrail check failed: {e}")
            # 出错时默认放行，避免阻断服务
            return {"next": "supervisor"}

    def _supervisor(self, state: AgentState):
        """监督员节点"""
        # (保持原有的 prompt 逻辑，但可以使用 prompt template 功能)
        system_prompt = (
            "你是 Pokemon 世界的顶尖博士助手 (Supervisor)，负责协调以下专家的工作：{members}\n\n"
            "专家职能：\n"
            "- chat：【日常闲聊与兜底】\n"
            "  • 处理问候、感谢等日常对话\n"
            "  • 当其他专家无法回答时进行尝试\n"
            "- kg_sqler：【知识图谱查询】\n"
            "  • 查询精确数据：身高体重、属性、特性效果\n"
            "  • 查询特定关系：某人的宝可梦、某地的道馆\n"
            "- graph_rager：【知识库RAG】\n"
            "  • 回答复杂问题：剧情背景、人物生平、传说故事\n"
            "- stats_agent：【数值与战斗专家】\n"
            "  • 分析宝可梦种族值强弱、对比两只宝可梦\n"
            "  • 计算属性克制关系、预测战斗胜负\n"
            "- pokedex_agent：【图鉴查询专家】\n"
            "  • 搜索宝可梦（按世代/属性）、查询进化链\n"
            "  • 查询招式列表、特性详细效果\n"
            "- trainer_agent：【训练师顾问】\n"
            "  • 组建队伍建议（雨天队/空间队等）、分析队伍打击面\n"
            "  • 推荐宝可梦配招（性格/努力值思路）\n"
            "- deep_researcher：【深度研究员】\n"
            "  • 执行复杂、多步骤的深度研究任务\n"
            "  • 生成详细的研究报告（如：进化历史、生态系统分析）\n"
            "- web_searcher：【情报探员】\n"
            "  • 查询最新发售信息、新闻、活动、当前对战环境 meta\n\n"
            "调度逻辑：\n"
            "1. 仔细分析用户意图，选择最匹配的专家。\n"
            "2. 优先使用 stats_agent 进行数值分析和对比。\n"
            "3. 优先使用 pokedex_agent 查询招式和进化。\n"
            "4. 优先使用 trainer_agent 进行配招和组队。\n"
            "4. 优先使用 trainer_agent 进行配招和组队。\n"
            "5. 对于需要深入调研、撰写报告的任务，使用 deep_researcher。\n"
            "6. kg_sqler 用于简单的实体属性查询。\n"
            "7. 涉及最新消息用 web_searcher。\n"
            "8. 任务完成返回 FINISH。\n"
        )

        prompt = ChatPromptTemplate.from_template("""
        {system_prompt}
        
        请严格按以下JSON格式回复:
        {{
            "next": "模块名称" (或 "FINISH")
        }}
        
        当前对话:
        {history}
        
        最新输入: {input}
        """)

        # 获取最后几条消息作为输入 context
        messages = state["messages"]
        
        # 提取最近几条历史作为 history 文本，避免传入过多 token
        history_msgs = messages[:-1]
        last_msg = messages[-1]
        
        history_text = "\n".join([f"{m.type}: {m.content}" for m in history_msgs[-5:]])
        
        chain = prompt | self.llm | JsonOutputParser()
        
        # 动态构建 members 描述
        members_desc = ", ".join(self.members)
        
        try:
            response = chain.invoke({
                "system_prompt": system_prompt.format(members=members_desc),
                "history": history_text,
                "input": last_msg.content
            })
            next_ = response.get("next", "FINISH")
        except Exception as e:
            logger.error(f"Supervisor parsing error: {e}")
            next_ = "FINISH"

        return {"next": END if next_ == "FINISH" else next_}

    def _chat(self, state: AgentState):
        """自然语言聊天节点"""
        # 修改为更智能的 Persona Prompt
        persona_prompt = (
            "你是一个热爱宝可梦的 AI 助手。你的名字叫 '洛托姆图鉴'。\n"
            "性格：热情、活泼、有时会加口癖 '洛托'。\n"
            "任务：回答用户关于宝可梦的问题，或者进行愉快的闲聊。\n"
            "知识截止：不要编造数据，如果不确定，请建议用户去查阅图鉴。\n"
            "注意：只回答宝可梦相关话题。如果用户强行聊无关话题，请委婉拒绝并拉回宝可梦话题。\n"
        )
        
        messages = state["messages"]
        # 在 messages 最前面插入 SystemMessage (如果还可以插入的话，LangGraph state 通常是 append only)
        # 这里我们临时构建一个 input 给 llm
        
        # 应用中间件
        context = MiddlewareContext(
            agent_name="chat_agent",
            thread_id=state.get("thread_id", ""),
            user_id=state.get("user_id", "")
        )
        messages = self.middleware.run_before_model(messages, context)
        
        # 注入 Persona
        if not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=persona_prompt)] + messages
        else:
            # 如果已有 SystemMessage，可能需要更新或保留
            pass 

        # 获取响应模式，默认为 text
        response_mode = state.get("response_mode", "text")
        
        try:
            if response_mode == "json":
                # 结构化输出模式
                structured_base_llm = self.base_llm.with_structured_output(AgentResponse)
                wrapped_structured_invoke = self.middleware.wrap_model_call(structured_base_llm.invoke, context)
                model_response = wrapped_structured_invoke(messages)
                
                if isinstance(model_response, AgentResponse):
                    content = model_response.model_dump_json()
                else:
                    import json
                    content = json.dumps(model_response, ensure_ascii=False) if isinstance(model_response, dict) else str(model_response)
                
                model_response = AIMessage(content=content)
            else:
                # 默认文本模式
                model_response = self.llm.invoke(messages)

        except Exception as e:
            logger.error(f"LLM invoke failed (mode={response_mode}): {e}")
            if response_mode == "json":
                from src.models.schemas import ErrorResponse
                error_resp = ErrorResponse(
                    error_code="llm_error", 
                    message=f"Failed to generate structured response: {str(e)}"
                )
                model_response = AIMessage(content=error_resp.model_dump_json())
            else:
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

    # [NEW] 子代理调用节点
    def _invoke_subagent(self, agent, state: AgentState, name: str):
        """通用子代理调用逻辑"""
        if not agent:
             return {"messages": [HumanMessage(content=f"{name} 未初始化", name=name)]}
        try:
             # 计算调用前的消息数量，用于提取新增消息
             start_len = len(state["messages"])
             
             # 调用子图
             result = agent.graph.invoke(state)
             
             # 获取结果消息
             all_msgs = result["messages"]
             
             # 为了避免消息重复 (LangGraph reducer)，我们只返回新增的消息
             new_msgs = all_msgs[start_len:]
             
             # 如果没有新消息（异常？），至少返回最后一条
             if not new_msgs and all_msgs:
                 new_msgs = [all_msgs[-1]]
             
             # 确保名字标记（可选）
             # for m in new_msgs:
             #    if not m.name: m.name = name
                 
             return {"messages": new_msgs}
        except Exception as e:
             logger.error(f"{name} invoke failed: {e}")
             return {"messages": [HumanMessage(content=f"{name} 运行失败: {e}", name=name)]}

    def _stats_node(self, state: AgentState):
        """数值分析节点"""
        return self._invoke_subagent(self.stats_agent, state, "stats_agent")

    def _pokedex_node(self, state: AgentState):
        """图鉴查询节点"""
        return self._invoke_subagent(self.pokedex_agent, state, "pokedex_agent")

    def _trainer_node(self, state: AgentState):
        """训练师助手节点"""
        return self._invoke_subagent(self.trainer_agent, state, "trainer_agent")

    def _deep_research_node(self, state: AgentState):
        """深度研究节点"""
        return self._invoke_subagent(self.deep_agent, state, "deep_researcher")

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

    # Time Travel APIs - 已由 BaseAgent 实现


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
