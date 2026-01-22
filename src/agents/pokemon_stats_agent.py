"""
PokemonStatsAgent - 宝可梦数据分析代理

功能:
- 分析宝可梦基础数值
- 对比两只宝可梦属性
- 计算属性克制关系
- 预测战斗结果
"""
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.prebuilt import ToolNode

from src.agents.base import ToolAgent
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Pokemon 属性克制表
TYPE_EFFECTIVENESS = {
    "普通": {"岩石": 0.5, "幽灵": 0, "钢": 0.5},
    "火": {"火": 0.5, "水": 0.5, "草": 2, "冰": 2, "虫": 2, "岩石": 0.5, "龙": 0.5, "钢": 2},
    "水": {"火": 2, "水": 0.5, "草": 0.5, "地面": 2, "岩石": 2, "龙": 0.5},
    "电": {"水": 2, "电": 0.5, "草": 0.5, "地面": 0, "飞行": 2, "龙": 0.5},
    "草": {"火": 0.5, "水": 2, "草": 0.5, "毒": 0.5, "地面": 2, "飞行": 0.5, "虫": 0.5, "岩石": 2, "龙": 0.5, "钢": 0.5},
    "冰": {"火": 0.5, "水": 0.5, "草": 2, "冰": 0.5, "地面": 2, "飞行": 2, "龙": 2, "钢": 0.5},
    "格斗": {"普通": 2, "冰": 2, "毒": 0.5, "飞行": 0.5, "超能力": 0.5, "虫": 0.5, "岩石": 2, "幽灵": 0, "恶": 2, "钢": 2, "妖精": 0.5},
    "毒": {"草": 2, "毒": 0.5, "地面": 0.5, "岩石": 0.5, "幽灵": 0.5, "钢": 0, "妖精": 2},
    "地面": {"火": 2, "电": 2, "草": 0.5, "毒": 2, "飞行": 0, "虫": 0.5, "岩石": 2, "钢": 2},
    "飞行": {"电": 0.5, "草": 2, "格斗": 2, "虫": 2, "岩石": 0.5, "钢": 0.5},
    "超能力": {"格斗": 2, "毒": 2, "超能力": 0.5, "恶": 0, "钢": 0.5},
    "虫": {"火": 0.5, "草": 2, "格斗": 0.5, "毒": 0.5, "飞行": 0.5, "超能力": 2, "幽灵": 0.5, "恶": 2, "钢": 0.5, "妖精": 0.5},
    "岩石": {"火": 2, "冰": 2, "格斗": 0.5, "地面": 0.5, "飞行": 2, "虫": 2, "钢": 0.5},
    "幽灵": {"普通": 0, "超能力": 2, "幽灵": 2, "恶": 0.5},
    "龙": {"龙": 2, "钢": 0.5, "妖精": 0},
    "恶": {"格斗": 0.5, "超能力": 2, "幽灵": 2, "恶": 0.5, "妖精": 0.5},
    "钢": {"火": 0.5, "水": 0.5, "电": 0.5, "冰": 2, "岩石": 2, "钢": 0.5, "妖精": 2},
    "妖精": {"火": 0.5, "格斗": 2, "毒": 0.5, "龙": 2, "恶": 2, "钢": 0.5},
}

# 示例宝可梦数据
POKEMON_DATA = {
    "皮卡丘": {"types": ["电"], "hp": 35, "attack": 55, "defense": 40, "sp_attack": 50, "sp_defense": 50, "speed": 90, "total": 320},
    "喷火龙": {"types": ["火", "飞行"], "hp": 78, "attack": 84, "defense": 78, "sp_attack": 109, "sp_defense": 85, "speed": 100, "total": 534},
    "水箭龟": {"types": ["水"], "hp": 79, "attack": 83, "defense": 100, "sp_attack": 85, "sp_defense": 105, "speed": 78, "total": 530},
    "妙蛙花": {"types": ["草", "毒"], "hp": 80, "attack": 82, "defense": 83, "sp_attack": 100, "sp_defense": 100, "speed": 80, "total": 525},
    "快龙": {"types": ["龙", "飞行"], "hp": 91, "attack": 134, "defense": 95, "sp_attack": 100, "sp_defense": 100, "speed": 80, "total": 600},
    "超梦": {"types": ["超能力"], "hp": 106, "attack": 110, "defense": 90, "sp_attack": 154, "sp_defense": 90, "speed": 130, "total": 680},
    "耿鬼": {"types": ["幽灵", "毒"], "hp": 60, "attack": 65, "defense": 60, "sp_attack": 130, "sp_defense": 75, "speed": 110, "total": 500},
    "卡比兽": {"types": ["普通"], "hp": 160, "attack": 110, "defense": 65, "sp_attack": 65, "sp_defense": 110, "speed": 30, "total": 540},
}


class PokemonNameSchema(BaseModel):
    pokemon_name: str = Field(description="宝可梦名称")


class ComparePokemonSchema(BaseModel):
    pokemon1: str = Field(description="第一只宝可梦名称")
    pokemon2: str = Field(description="第二只宝可梦名称")


class TypeEffectivenessSchema(BaseModel):
    attack_type: str = Field(description="攻击属性")
    defend_types: List[str] = Field(description="防御方属性列表")


@tool(args_schema=PokemonNameSchema)
def analyze_pokemon_stats(pokemon_name: str) -> str:
    """分析指定宝可梦的基础数值和属性"""
    if pokemon_name not in POKEMON_DATA:
        return f"未找到宝可梦: {pokemon_name}"
    
    data = POKEMON_DATA[pokemon_name]
    result = f"""
## {pokemon_name} 数据分析

**属性**: {', '.join(data['types'])}

**基础数值**:
- HP: {data['hp']}
- 攻击: {data['attack']}
- 防御: {data['defense']}
- 特攻: {data['sp_attack']}
- 特防: {data['sp_defense']}
- 速度: {data['speed']}
- **种族值总和**: {data['total']}

**评价**:
- 最高属性: {max(['hp', 'attack', 'defense', 'sp_attack', 'sp_defense', 'speed'], key=lambda x: data[x])}
- 最低属性: {min(['hp', 'attack', 'defense', 'sp_attack', 'sp_defense', 'speed'], key=lambda x: data[x])}
"""
    return result


@tool(args_schema=ComparePokemonSchema)
def compare_pokemon(pokemon1: str, pokemon2: str) -> str:
    """对比两只宝可梦的属性"""
    if pokemon1 not in POKEMON_DATA:
        return f"未找到宝可梦: {pokemon1}"
    if pokemon2 not in POKEMON_DATA:
        return f"未找到宝可梦: {pokemon2}"
    
    d1, d2 = POKEMON_DATA[pokemon1], POKEMON_DATA[pokemon2]
    
    comparison = f"""
## {pokemon1} vs {pokemon2} 对比

| 属性 | {pokemon1} | {pokemon2} | 优势 |
|------|------------|------------|------|
| HP | {d1['hp']} | {d2['hp']} | {'←' if d1['hp'] > d2['hp'] else '→' if d1['hp'] < d2['hp'] else '='} |
| 攻击 | {d1['attack']} | {d2['attack']} | {'←' if d1['attack'] > d2['attack'] else '→' if d1['attack'] < d2['attack'] else '='} |
| 防御 | {d1['defense']} | {d2['defense']} | {'←' if d1['defense'] > d2['defense'] else '→' if d1['defense'] < d2['defense'] else '='} |
| 特攻 | {d1['sp_attack']} | {d2['sp_attack']} | {'←' if d1['sp_attack'] > d2['sp_attack'] else '→' if d1['sp_attack'] < d2['sp_attack'] else '='} |
| 特防 | {d1['sp_defense']} | {d2['sp_defense']} | {'←' if d1['sp_defense'] > d2['sp_defense'] else '→' if d1['sp_defense'] < d2['sp_defense'] else '='} |
| 速度 | {d1['speed']} | {d2['speed']} | {'←' if d1['speed'] > d2['speed'] else '→' if d1['speed'] < d2['speed'] else '='} |
| **总和** | **{d1['total']}** | **{d2['total']}** | **{'←' if d1['total'] > d2['total'] else '→' if d1['total'] < d2['total'] else '='}** |

**属性**: {pokemon1}({', '.join(d1['types'])}) vs {pokemon2}({', '.join(d2['types'])})
"""
    return comparison


@tool(args_schema=TypeEffectivenessSchema)
def type_effectiveness(attack_type: str, defend_types: List[str]) -> str:
    """计算属性克制关系"""
    if attack_type not in TYPE_EFFECTIVENESS:
        return f"未知攻击属性: {attack_type}"
    
    multiplier = 1.0
    for def_type in defend_types:
        if def_type in TYPE_EFFECTIVENESS.get(attack_type, {}):
            multiplier *= TYPE_EFFECTIVENESS[attack_type][def_type]
    
    effectiveness = "普通"
    if multiplier == 0:
        effectiveness = "无效 (0x)"
    elif multiplier < 1:
        effectiveness = f"效果不好 ({multiplier}x)"
    elif multiplier > 1:
        effectiveness = f"效果拔群! ({multiplier}x)"
    else:
        effectiveness = "普通 (1x)"
    
    return f"**{attack_type}** 属性攻击 **{'/'.join(defend_types)}** 属性: {effectiveness}"


@tool(args_schema=ComparePokemonSchema)
def battle_prediction(pokemon1: str, pokemon2: str) -> str:
    """预测两只宝可梦战斗结果"""
    if pokemon1 not in POKEMON_DATA or pokemon2 not in POKEMON_DATA:
        return "需要两只有效的宝可梦进行对战预测"
    
    d1, d2 = POKEMON_DATA[pokemon1], POKEMON_DATA[pokemon2]
    
    # 简单预测逻辑: 考虑种族值和属性克制
    score1, score2 = d1['total'], d2['total']
    
    # 属性克制加成
    for t1 in d1['types']:
        for t2 in d2['types']:
            if t2 in TYPE_EFFECTIVENESS.get(t1, {}):
                score1 += 50 * TYPE_EFFECTIVENESS[t1][t2]
    
    for t2 in d2['types']:
        for t1 in d1['types']:
            if t1 in TYPE_EFFECTIVENESS.get(t2, {}):
                score2 += 50 * TYPE_EFFECTIVENESS[t2][t1]
    
    # 速度优势
    if d1['speed'] > d2['speed']:
        score1 += 30
    elif d2['speed'] > d1['speed']:
        score2 += 30
    
    winner = pokemon1 if score1 > score2 else pokemon2 if score2 > score1 else "平局"
    confidence = abs(score1 - score2) / max(score1, score2) * 100
    
    return f"""
## 对战预测: {pokemon1} vs {pokemon2}

**预测胜者**: {winner}
**信心指数**: {confidence:.1f}%

**分析**:
- {pokemon1} 综合评分: {score1:.0f}
- {pokemon2} 综合评分: {score2:.0f}
- 速度优势: {'先手 ' + pokemon1 if d1['speed'] > d2['speed'] else '先手 ' + pokemon2}
"""


class PokemonStatsAgent(ToolAgent):
    """宝可梦数据分析代理"""
    
    def __init__(self):
        tools = [analyze_pokemon_stats, compare_pokemon, type_effectiveness, battle_prediction]
        super().__init__(tools=tools, bind_tools=True)
        self.tool_node = ToolNode(self._tools)
        logger.info("PokemonStatsAgent initialized")

    def get_info(self) -> dict:
        return {
            "name": "PokemonStatsAgent",
            "description": "宝可梦数据分析代理 - 属性对比、克制关系、战斗预测",
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
                new_messages.append({
                    "role": "tool",
                    "name": call["name"],
                    "content": result,
                    "tool_call_id": call["id"]
                })
        return {"messages": new_messages}

    def _build_graph(self):
        workflow = StateGraph(MessagesState)
        workflow.add_node("agent", self._call_model)
        workflow.add_node("run_tool", self._run_tool)

        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges("agent", self._should_continue, {
            "run_tool": "run_tool",
            "end": END
        })
        workflow.add_edge("run_tool", "agent")

        return workflow.compile(checkpointer=self.checkpointer)


if __name__ == "__main__":
    agent = PokemonStatsAgent()
    print(analyze_pokemon_stats.invoke({"pokemon_name": "皮卡丘"}))
    print(compare_pokemon.invoke({"pokemon1": "喷火龙", "pokemon2": "水箭龟"}))
    print(type_effectiveness.invoke({"attack_type": "水", "defend_types": ["火"]}))
    print(battle_prediction.invoke({"pokemon1": "超梦", "pokemon2": "快龙"}))
