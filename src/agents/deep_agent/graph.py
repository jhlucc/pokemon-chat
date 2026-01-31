"""
Pokemon Deep Research Agent

基于 LangGraph 1.x 实现的 Pokemon 深度研究代理。
参考 open-deep-research 项目的迭代研究模式。

工作流程:
1. 生成研究查询 (Generate Queries)
2. 搜索知识库 (Search Knowledge Base)
3. 处理学习结果 (Process Learnings)
4. 决定是否深入 (Route: Continue/Finalize)
5. 生成最终报告 (Generate Report)
"""

from typing import Any, Literal

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from src.agents.base import BaseAgent
from src.agents.deep_agent.context import DeepContext
from src.agents.pokemon_data import get_pokemon_data
from src.agents.pokemon_entities import extract_pokemon_entities
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ============== Helpers ==============


def _is_noneish(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and value.strip().lower() in {"", "none", "null", "nan"}:
        return True
    return False


def _as_list(value: Any) -> list[str]:
    if _is_noneish(value):
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if not _is_noneish(v) and str(v).strip()]
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    return [str(value).strip()] if str(value).strip() else []


def _facts_for_pokemon(cn_name: str) -> list[str]:
    """Deterministic facts from local dataset (no LLM)."""
    data = get_pokemon_data()
    rec = data.get_by_cn_name(cn_name)
    if not rec:
        return []

    facts: list[str] = []
    types = _as_list(rec.get("type"))
    if types:
        facts.append(f"{cn_name} 属性: {'/'.join(types)}")

    abilities = _as_list(rec.get("ability"))
    if abilities:
        facts.append(f"{cn_name} 特性: {', '.join(abilities)}")

    hidden = _as_list(rec.get("隐藏特性"))
    if hidden:
        facts.append(f"{cn_name} 隐藏特性: {', '.join(hidden)}")

    evo = rec.get("进化")
    if isinstance(evo, str) and not _is_noneish(evo):
        facts.append(f"{cn_name} 可进化为: {evo.strip()}")

    return facts


# ============== Graph Nodes ==============


async def generate_queries_node(state: DeepContext, config: RunnableConfig) -> dict[str, Any]:
    """生成研究查询 - 基于当前主题和已有学习"""
    topic = state.get("topic", "")

    # [NEW] 如果没有 topic，尝试从消息中提取
    if not topic and state.get("messages"):
        last_msg = state["messages"][-1]
        topic = last_msg.content
        logger.info(f"[DeepResearch] 从消息提取主题: {topic}")

    state.get("learnings", [])
    breadth = state.get("breadth", 3)

    logger.info(f"[DeepResearch] 生成查询 | Topic: {topic} | Breadth: {breadth}")

    # 模拟生成研究方向 (实际应用中由 LLM 生成)
    research_directions = []
    if "皮卡丘" in topic or "电" in topic:
        research_directions = ["皮卡丘的进化条件", "电属性克制关系", "皮卡丘的招式学习"]
    elif "对战" in topic or "战斗" in topic:
        research_directions = ["属性克制分析", "队伍构建策略", "最强宝可梦排行"]
    else:
        research_directions = ["宝可梦基础属性", "进化链研究", "属性相克关系"]

    return {
        "topic": topic,  # 以此确保 topic 被回写到状态中
        "research_directions": research_directions[:breadth],
        "messages": [AIMessage(content=f"生成了 {len(research_directions[:breadth])} 个研究方向")],
    }


async def search_knowledge_node(state: DeepContext, config: RunnableConfig) -> dict[str, Any]:
    """搜索知识库 - 从 Pokemon 知识图谱检索信息"""
    directions = state.get("research_directions", [])
    current_learnings = state.get("learnings", [])
    pokemon_entities = state.get("pokemon_entities", [])

    logger.info(f"[DeepResearch] 搜索知识库 | Directions: {directions}")

    new_learnings: list[str] = []
    new_entities: list[str] = []

    # Use local dataset as the first truth source.
    topic = state.get("topic", "")
    candidates: list[str] = []
    candidates.extend(extract_pokemon_entities(topic))
    for d in directions:
        candidates.extend(extract_pokemon_entities(d))
    candidates.extend([e for e in pokemon_entities if isinstance(e, str)])

    # De-dupe while keeping order.
    seen: set[str] = set()
    ordered_candidates: list[str] = []
    for c in candidates:
        if c in seen:
            continue
        seen.add(c)
        ordered_candidates.append(c)

    for pokemon in ordered_candidates:
        facts = _facts_for_pokemon(pokemon)
        if facts:
            new_learnings.extend(facts[:4])
            if pokemon not in pokemon_entities:
                new_entities.append(pokemon)

    # 去重
    all_learnings = list(set(current_learnings + new_learnings))
    all_entities = list(set(pokemon_entities + new_entities))

    return {
        "learnings": all_learnings,
        "pokemon_entities": all_entities,
        "messages": [AIMessage(content=f"发现 {len(new_learnings)} 条新知识, {len(new_entities)} 个宝可梦实体")],
    }


async def process_learnings_node(state: DeepContext, config: RunnableConfig) -> dict[str, Any]:
    """处理学习结果 - 分析并生成后续研究方向"""
    learnings = state.get("learnings", [])
    iterations = state.get("iterations", 0)

    logger.info(f"[DeepResearch] 处理学习结果 | Iteration: {iterations} | Learnings: {len(learnings)}")

    # 生成后续研究问题
    follow_up = []
    if any("进化" in learning for learning in learnings):
        follow_up.append("进化条件研究")
    if any("属性" in learning for learning in learnings):
        follow_up.append("属性克制深入分析")
    if any("招式" in learning for learning in learnings):
        follow_up.append("最优配招研究")

    return {
        "research_directions": follow_up,
        "iterations": iterations + 1,
        "messages": [AIMessage(content=f"迭代 {iterations + 1}: 生成 {len(follow_up)} 个后续研究方向")],
    }


def route_decision(state: DeepContext) -> Literal["continue_research", "finalize"]:
    """决定是否继续深入研究"""
    iterations = state.get("iterations", 0)
    max_depth = state.get("max_depth", 2)
    learnings = state.get("learnings", [])

    # 停止条件: 达到最大深度 或 学习内容足够丰富
    if iterations >= max_depth or len(learnings) >= 10:
        logger.info(f"[DeepResearch] 研究完成 | Iterations: {iterations} | Learnings: {len(learnings)}")
        return "finalize"

    logger.info(f"[DeepResearch] 继续深入 | Iterations: {iterations}/{max_depth}")
    return "continue_research"


async def finalize_report_node(state: DeepContext, config: RunnableConfig) -> dict[str, Any]:
    """生成最终研究报告"""
    topic = state.get("topic", "Pokemon 研究")
    learnings = state.get("learnings", [])
    pokemon_entities = state.get("pokemon_entities", [])
    iterations = state.get("iterations", 0)

    logger.info(f"[DeepResearch] 生成报告 | Topic: {topic}")

    # 构建 Markdown 报告
    report = f"""# Pokemon 深度研究报告: {topic}

## 研究概述
- **研究迭代次数**: {iterations}
- **发现的宝可梦实体**: {", ".join(pokemon_entities) if pokemon_entities else "无"}

## 研究发现

"""
    for i, learning in enumerate(learnings, 1):
        report += f"{i}. {learning}\n"

    report += f"""
## 研究结论

基于 {len(learnings)} 条知识点的分析，我们对 **{topic}** 有了深入了解。
涉及的宝可梦包括: {", ".join(pokemon_entities)}。

---
*本报告由 Pokemon Deep Research Agent 自动生成*
"""

    return {"final_report": report, "messages": [AIMessage(content="研究报告生成完成")]}


# ============== DeepAgent Class ==============


class DeepAgent(BaseAgent):
    """
    Pokemon Deep Research Agent

    实现迭代式深度研究:
    生成查询 → 搜索知识库 → 处理学习 → 决策(继续/完成) → 生成报告
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        logger.info("DeepAgent (Pokemon Research) initialized")

    def _build_graph(self):
        """构建 LangGraph 工作流"""
        builder = StateGraph(DeepContext)

        # 添加节点
        builder.add_node("generate_queries", generate_queries_node)
        builder.add_node("search_knowledge", search_knowledge_node)
        builder.add_node("process_learnings", process_learnings_node)
        builder.add_node("finalize_report", finalize_report_node)

        # 定义边
        builder.add_edge(START, "generate_queries")
        builder.add_edge("generate_queries", "search_knowledge")
        builder.add_edge("search_knowledge", "process_learnings")

        # 条件路由
        builder.add_conditional_edges(
            "process_learnings",
            route_decision,
            {
                "continue_research": "generate_queries",  # 循环回去继续研究
                "finalize": "finalize_report",
            },
        )

        builder.add_edge("finalize_report", END)

        return builder.compile()

    def get_info(self) -> dict:
        return {"name": "DeepAgent", "description": "Pokemon 深度研究代理 - 迭代式多轮研究，自动生成研究报告"}

    async def research(self, topic: str, breadth: int = 3, max_depth: int = 2, **kwargs) -> str:
        """
        执行深度研究

        Args:
            topic: 研究主题
            breadth: 每轮生成的查询数量
            max_depth: 最大研究深度

        Returns:
            Markdown 格式的研究报告
        """
        initial_state: DeepContext = {
            "topic": topic,
            "messages": [HumanMessage(content=topic)],
            "iterations": 0,
            "breadth": breadth,
            "depth": 0,
            "max_depth": max_depth,
            "learnings": [],
            "research_directions": [],
            "sources": [],
            "pokemon_entities": [],
            "type_analysis": {},
            "battle_insights": [],
            "final_report": None,
        }

        config = {"configurable": {"thread_id": kwargs.get("thread_id", "deep_research")}}

        # 运行图
        final_state = None
        async for state in self.graph.astream(initial_state, config):
            final_state = state

        # 获取最终报告
        if final_state and "finalize_report" in final_state:
            return final_state["finalize_report"].get("final_report", "研究失败")

        return "研究未完成"

    async def query(self, question: str, **kwargs) -> str:
        """兼容接口"""
        return await self.research(question, **kwargs)


# ============== Test ==============

if __name__ == "__main__":
    import asyncio

    async def test():
        agent = DeepAgent()
        report = await agent.research("皮卡丘的进化和战斗能力", breadth=2, max_depth=2)
        print(report)

    asyncio.run(test())
