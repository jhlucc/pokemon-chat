"""
PokedexAgent - 宝可梦图鉴查询代理

功能:
- 搜索宝可梦
- 获取进化链
- 查询可学招式
- 查询特性效果
"""

from langchain_core.tools import tool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field

from src.agents.base import ToolAgent
from src.utils.logger import get_logger

logger = get_logger(__name__)

# 宝可梦图鉴数据
POKEDEX = {
    "001": {
        "name": "妙蛙种子",
        "types": ["草", "毒"],
        "generation": 1,
        "category": "种子宝可梦",
        "height": 0.7,
        "weight": 6.9,
    },
    "004": {"name": "小火龙", "types": ["火"], "generation": 1, "category": "蜥蜴宝可梦", "height": 0.6, "weight": 8.5},
    "007": {"name": "杰尼龟", "types": ["水"], "generation": 1, "category": "小龟宝可梦", "height": 0.5, "weight": 9.0},
    "025": {"name": "皮卡丘", "types": ["电"], "generation": 1, "category": "鼠宝可梦", "height": 0.4, "weight": 6.0},
    "143": {
        "name": "卡比兽",
        "types": ["普通"],
        "generation": 1,
        "category": "睡觉宝可梦",
        "height": 2.1,
        "weight": 460.0,
    },
    "150": {
        "name": "超梦",
        "types": ["超能力"],
        "generation": 1,
        "category": "基因宝可梦",
        "height": 2.0,
        "weight": 122.0,
    },
    "149": {
        "name": "快龙",
        "types": ["龙", "飞行"],
        "generation": 1,
        "category": "龙宝可梦",
        "height": 2.2,
        "weight": 210.0,
    },
    "094": {
        "name": "耿鬼",
        "types": ["幽灵", "毒"],
        "generation": 1,
        "category": "影子宝可梦",
        "height": 1.5,
        "weight": 40.5,
    },
}

# 进化链
EVOLUTION_CHAINS = {
    "妙蛙种子": ["妙蛙种子", "妙蛙草", "妙蛙花"],
    "小火龙": ["小火龙", "火恐龙", "喷火龙"],
    "杰尼龟": ["杰尼龟", "卡咪龟", "水箭龟"],
    "皮卡丘": ["皮丘", "皮卡丘", "雷丘"],
    "迷你龙": ["迷你龙", "哈克龙", "快龙"],
    "鬼斯": ["鬼斯", "鬼斯通", "耿鬼"],
}

# 招式数据
MOVES = {
    "皮卡丘": ["电击", "十万伏特", "电光一闪", "铁尾", "打雷"],
    "喷火龙": ["火焰拳", "喷射火焰", "龙之怒", "空气斩", "大字爆炎"],
    "水箭龟": ["水枪", "水炮", "火箭头槌", "冲浪", "贝壳夹击"],
    "超梦": ["精神强念", "幻象光线", "自我再生", "冥想", "精神破坏"],
    "快龙": ["龙之怒", "逆鳞", "暴风", "神速", "龙之舞"],
}

# 特性数据
ABILITIES = {
    "静电": {"description": "接触到自己的对手可能会陷入麻痹状态", "pokemon": ["皮卡丘"]},
    "猛火": {"description": "HP剩余量少时，火属性招式的威力提高", "pokemon": ["小火龙", "喷火龙"]},
    "激流": {"description": "HP剩余量少时，水属性招式的威力提高", "pokemon": ["杰尼龟", "水箭龟"]},
    "茂盛": {"description": "HP剩余量少时，草属性招式的威力提高", "pokemon": ["妙蛙种子", "妙蛙花"]},
    "压迫感": {"description": "给对手带来压迫感，大量减少其使用招式的PP", "pokemon": ["超梦"]},
    "精神力": {"description": "精神力强，不会畏缩", "pokemon": ["快龙"]},
}


class SearchSchema(BaseModel):
    query: str = Field(description="搜索关键词(名称/属性/世代)")


class PokemonNameSchema(BaseModel):
    pokemon_name: str = Field(description="宝可梦名称")


class AbilitySchema(BaseModel):
    ability_name: str = Field(description="特性名称")


@tool(args_schema=SearchSchema)
def search_pokedex(query: str) -> str:
    """按名称、属性或世代搜索宝可梦"""
    results = []
    query.lower()

    for pid, data in POKEDEX.items():
        # 名称匹配
        if query in data["name"]:
            results.append(f"#{pid} {data['name']} ({'/'.join(data['types'])}) - {data['category']}")
        # 属性匹配
        elif query in data["types"]:
            results.append(f"#{pid} {data['name']} ({'/'.join(data['types'])}) - {data['category']}")
        # 世代匹配
        elif query.isdigit() and int(query) == data["generation"]:
            results.append(f"#{pid} {data['name']} ({'/'.join(data['types'])}) - 第{data['generation']}世代")

    if not results:
        return f"未找到匹配 '{query}' 的宝可梦"

    return f"**搜索结果 ({len(results)})**:\n" + "\n".join(results)


@tool(args_schema=PokemonNameSchema)
def get_evolution_chain(pokemon_name: str) -> str:
    """获取宝可梦的进化链"""
    for _starter, chain in EVOLUTION_CHAINS.items():
        if pokemon_name in chain:
            index = chain.index(pokemon_name)
            chain_display = " → ".join([f"**{p}**" if p == pokemon_name else p for p in chain])
            return f"**{pokemon_name}** 的进化链:\n{chain_display}\n\n当前为第 {index + 1}/{len(chain)} 阶段"

    return f"{pokemon_name} 没有进化链信息(可能是无法进化的宝可梦)"


@tool(args_schema=PokemonNameSchema)
def get_pokemon_moves(pokemon_name: str) -> str:
    """获取宝可梦可学习的招式"""
    if pokemon_name not in MOVES:
        return f"未找到 {pokemon_name} 的招式信息"

    move_list = MOVES[pokemon_name]
    return f"**{pokemon_name}** 可学习的招式:\n" + "\n".join([f"- {move}" for move in move_list])


@tool(args_schema=AbilitySchema)
def get_ability_info(ability_name: str) -> str:
    """查询特性效果"""
    if ability_name not in ABILITIES:
        return f"未找到特性: {ability_name}"

    ability = ABILITIES[ability_name]
    pokemon_list = ", ".join(ability["pokemon"])
    return f"""**特性: {ability_name}**

**效果**: {ability["description"]}

**拥有此特性的宝可梦**: {pokemon_list}
"""


class PokedexAgent(ToolAgent):
    """宝可梦图鉴查询代理"""

    def __init__(self):
        tools = [search_pokedex, get_evolution_chain, get_pokemon_moves, get_ability_info]
        super().__init__(tools=tools, bind_tools=True)
        self.tool_node = ToolNode(self._tools)
        logger.info("PokedexAgent initialized")

    def get_info(self) -> dict:
        return {
            "name": "PokedexAgent",
            "description": "宝可梦图鉴查询代理 - 搜索宝可梦、进化链、招式、特性",
            "tools": [t.name for t in self._tools],
        }

    def _call_model(self, state):
        messages = state["messages"]
        response = self.llm_with_tools.invoke(messages)
        return {"messages": [response]}

    def _should_continue(self, state):
        last_msg = state["messages"][-1]
        if not last_msg.tool_calls:
            return "end"
        return "run_tool"

    def _run_tool(self, state):
        new_messages = []
        tool_calls = state["messages"][-1].tool_calls
        tool_map = {t.name: t for t in self._tools}

        for call in tool_calls:
            tool = tool_map.get(call["name"])
            if tool:
                result = tool.invoke(call["args"])
                new_messages.append(
                    {"role": "tool", "name": call["name"], "content": result, "tool_call_id": call["id"]}
                )
        return {"messages": new_messages}

    def _build_graph(self):
        workflow = StateGraph(MessagesState)
        workflow.add_node("agent", self._call_model)
        workflow.add_node("run_tool", self._run_tool)

        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges("agent", self._should_continue, {"run_tool": "run_tool", "end": END})
        workflow.add_edge("run_tool", "agent")

        return workflow.compile(checkpointer=self.checkpointer)


if __name__ == "__main__":
    print(search_pokedex.invoke({"query": "电"}))
    print(get_evolution_chain.invoke({"pokemon_name": "皮卡丘"}))
    print(get_pokemon_moves.invoke({"pokemon_name": "喷火龙"}))
    print(get_ability_info.invoke({"ability_name": "静电"}))
