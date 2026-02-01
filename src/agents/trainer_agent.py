"""
TrainerAgent - 训练师助手代理

功能:
- 队伍构建建议
- 克制建议
- 属性覆盖分析
- 配招推荐
"""

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from src.agents.base import ToolAgent
from src.utils.logger import get_logger

logger = get_logger(__name__)

# 属性列表
ALL_TYPES = [
    "普通",
    "火",
    "水",
    "电",
    "草",
    "冰",
    "格斗",
    "毒",
    "地面",
    "飞行",
    "超能力",
    "虫",
    "岩石",
    "幽灵",
    "龙",
    "恶",
    "钢",
    "妖精",
]

# 推荐队伍模板
TEAM_TEMPLATES = {
    "平衡队": {
        "description": "全面平衡的队伍，覆盖多种属性",
        "members": ["皮卡丘", "喷火龙", "水箭龟", "妙蛙花", "快龙", "卡比兽"],
        "coverage": ["电", "火", "飞行", "水", "草", "毒", "龙", "普通"],
    },
    "速攻队": {
        "description": "高速扫荡队伍",
        "members": ["皮卡丘", "快龙", "超梦", "耿鬼", "喷火龙", "雷丘"],
        "coverage": ["电", "龙", "飞行", "超能力", "幽灵", "毒", "火"],
    },
    "耐久队": {
        "description": "高耐久防守队伍",
        "members": ["卡比兽", "水箭龟", "妙蛙花", "钢铠鸦", "幸福蛋", "盔甲鸟"],
        "coverage": ["普通", "水", "草", "毒", "钢", "飞行"],
    },
}

# 经典配招
MOVESETS = {
    "皮卡丘": {
        "物攻型": ["十万伏特", "铁尾", "电光一闪", "打雷"],
        "辅助型": ["电击", "电磁波", "光墙", "替身"],
    },
    "喷火龙": {
        "特攻型": ["喷射火焰", "大字爆炎", "空气斩", "龙之波动"],
        "剑舞型": ["剑舞", "火焰拳", "地震", "逆鳞"],
    },
    "水箭龟": {
        "炮台型": ["水炮", "冰冻光束", "恶之波动", "气合弹"],
        "物耐型": ["冲浪", "地震", "火箭头槌", "铁壁"],
    },
    "超梦": {
        "特攻型": ["精神强念", "气合弹", "暗影球", "冰冻光束"],
        "冥想型": ["冥想", "精神强念", "自我再生", "气合弹"],
    },
    "快龙": {
        "龙舞型": ["龙之舞", "逆鳞", "地震", "神速"],
        "混合型": ["流星群", "火焰喷射", "逆鳞", "神速"],
    },
}

# 属性克制
TYPE_WEAKNESSES = {
    "火": ["水", "地面", "岩石"],
    "水": ["电", "草"],
    "草": ["火", "冰", "毒", "飞行", "虫"],
    "电": ["地面"],
    "冰": ["火", "格斗", "岩石", "钢"],
    "龙": ["冰", "龙", "妖精"],
    "超能力": ["虫", "幽灵", "恶"],
    "普通": ["格斗"],
    "幽灵": ["幽灵", "恶"],
}


class TeamRequestSchema(BaseModel):
    style: str = Field(description="队伍风格(平衡队/速攻队/耐久队)")


class CounterSchema(BaseModel):
    opponent_types: list[str] = Field(description="对手队伍的属性列表")


class CoverageSchema(BaseModel):
    team_types: list[str] = Field(description="队伍成员的属性列表")


class MovesetSchema(BaseModel):
    pokemon_name: str = Field(description="宝可梦名称")


@tool(args_schema=TeamRequestSchema)
def build_team(style: str) -> str:
    """根据风格推荐队伍"""
    if style not in TEAM_TEMPLATES:
        available = ", ".join(TEAM_TEMPLATES.keys())
        return f"未知的队伍风格: {style}\n可用风格: {available}"

    team = TEAM_TEMPLATES[style]
    members = "\n".join([f"- {m}" for m in team["members"]])
    coverage = ", ".join(team["coverage"])

    return f"""## 推荐队伍: {style}

**描述**: {team["description"]}

**队伍成员**:
{members}

**属性覆盖**: {coverage}
"""


@tool(args_schema=CounterSchema)
def counter_team(opponent_types: list[str]) -> str:
    """针对对手队伍给出克制建议"""
    counters = {}

    for opp_type in opponent_types:
        if opp_type in TYPE_WEAKNESSES:
            for weakness in TYPE_WEAKNESSES[opp_type]:
                counters[weakness] = counters.get(weakness, 0) + 1

    if not counters:
        return "无法分析克制关系，请确认属性名称正确"

    sorted_counters = sorted(counters.items(), key=lambda x: -x[1])

    result = f"## 针对 {'/'.join(opponent_types)} 的克制建议\n\n"
    result += "**推荐携带属性**:\n"
    for type_name, count in sorted_counters[:5]:
        result += f"- **{type_name}** (克制 {count} 个属性)\n"

    return result


@tool(args_schema=CoverageSchema)
def type_coverage(team_types: list[str]) -> str:
    """分析队伍属性覆盖"""
    team_types = [t.strip() for t in team_types if isinstance(t, str) and t.strip()]
    unknown_types = [t for t in team_types if t not in ALL_TYPES]
    known_types = [t for t in team_types if t in ALL_TYPES]

    covered = set()

    # 简化的克制表
    coverage_map = {
        "火": ["草", "冰", "虫", "钢"],
        "水": ["火", "地面", "岩石"],
        "草": ["水", "地面", "岩石"],
        "电": ["水", "飞行"],
        "冰": ["草", "地面", "飞行", "龙"],
        "格斗": ["普通", "冰", "岩石", "恶", "钢"],
        "地面": ["火", "电", "毒", "岩石", "钢"],
        "飞行": ["草", "格斗", "虫"],
        "超能力": ["格斗", "毒"],
        "龙": ["龙"],
        "恶": ["超能力", "幽灵"],
        "钢": ["冰", "岩石", "妖精"],
        "妖精": ["格斗", "龙", "恶"],
    }

    for t in known_types:
        if t in coverage_map:
            covered.update(coverage_map[t])

    uncovered = set(ALL_TYPES) - covered

    coverage_pct = len(covered) / len(ALL_TYPES) * 100

    return f"""## 属性覆盖分析

**队伍属性**: {", ".join(team_types) if team_types else "无"}
{f"**未知属性**: {', '.join(unknown_types)}" if unknown_types else ""}

**可有效打击** ({len(covered)}/{len(ALL_TYPES)}):
{", ".join(sorted(covered)) if covered else "无"}

**无法有效打击** ({len(uncovered)}):
{", ".join(sorted(uncovered)) if uncovered else "无"}

**覆盖率**: {coverage_pct:.1f}%
"""


@tool(args_schema=MovesetSchema)
def suggest_moveset(pokemon_name: str) -> str:
    """推荐配招"""
    if pokemon_name not in MOVESETS:
        return f"未找到 {pokemon_name} 的配招建议"

    sets = MOVESETS[pokemon_name]
    result = f"## {pokemon_name} 推荐配招\n\n"

    for set_name, moves in sets.items():
        result += f"### {set_name}\n"
        result += "\n".join([f"- {move}" for move in moves])
        result += "\n\n"

    return result


class TrainerAgent(ToolAgent):
    """训练师助手代理 - 队伍构建、克制建议、配招推荐"""

    def __init__(self):
        tools = [build_team, counter_team, type_coverage, suggest_moveset]
        super().__init__(tools=tools, bind_tools=True)
        logger.info("TrainerAgent initialized")

    def get_info(self) -> dict:
        return {
            "name": "TrainerAgent",
            "description": "训练师助手代理 - 队伍构建、克制建议、配招推荐",
            "tools": [t.name for t in self._tools],
        }

    # 使用 ToolAgent 基类的默认实现:
    # - _build_graph()
    # - _call_model()
    # - _should_continue()
    # - _run_tool()


if __name__ == "__main__":
    print(build_team.invoke({"style": "平衡队"}))
    print(counter_team.invoke({"opponent_types": ["火", "龙"]}))
    print(type_coverage.invoke({"team_types": ["水", "电", "草"]}))
    print(suggest_moveset.invoke({"pokemon_name": "喷火龙"}))
