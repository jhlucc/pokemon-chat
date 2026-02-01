"""
PokedexAgent - 宝可梦图鉴查询代理

功能:
- 搜索宝可梦
- 获取进化链
- 查询可学招式
- 查询特性效果
"""

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from src.agents.base import ToolAgent
from src.agents.pokemon_data import get_pokemon_data
from src.agents.pokemon_facts import evolution_chain
from src.utils.logger import get_logger

logger = get_logger(__name__)

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
    q = (query or "").strip()
    if not q:
        return "请输入搜索关键词（宝可梦名称 / 属性 / 编号）"

    data = get_pokemon_data()
    results: list[tuple[int, str]] = []  # (id, formatted)

    # 1) Exact-ish name match (CN/EN/JP)
    resolved = data.resolve_name(q)
    if resolved:
        rec = data.get_by_cn_name(resolved)
        if rec:
            pid = rec.get("id")
            pid_int = int(pid) if isinstance(pid, int) else None
            types = rec.get("type") or []
            types_str = "/".join(types) if isinstance(types, list) else str(types)
            line = f"#{pid_int:03d} {resolved} ({types_str})" if isinstance(pid_int, int) else f"{resolved} ({types_str})"
            results.append((pid_int or 0, line))

    # 2) Numeric id match
    if q.isdigit():
        rec = data.get_by_id(q)
        if rec:
            name = rec.get("chinese_name") or ""
            pid = rec.get("id")
            pid_int = int(pid) if isinstance(pid, int) else None
            types = rec.get("type") or []
            types_str = "/".join(types) if isinstance(types, list) else str(types)
            line = f"#{pid_int:03d} {name} ({types_str})" if isinstance(pid_int, int) else f"{name} ({types_str})"
            results.append((pid_int or 0, line))

    # 3) Fuzzy search over dataset (type match / substring name match)
    q_lower = q.lower()
    for rec in data.iter_all():
        name = rec.get("chinese_name") or ""
        if not isinstance(name, str) or not name:
            continue
        en = rec.get("english_name") or ""
        jp = rec.get("japanese_name") or ""
        types = rec.get("type") or []

        hit = False
        if isinstance(types, list) and q in types:
            hit = True
        elif isinstance(name, str) and q in name:
            hit = True
        elif isinstance(en, str) and en and q_lower in en.lower():
            hit = True
        elif isinstance(jp, str) and jp and q in jp:
            hit = True

        if not hit:
            continue

        pid = rec.get("id")
        pid_int = int(pid) if isinstance(pid, int) else None
        types_str = "/".join(types) if isinstance(types, list) else str(types)
        line = f"#{pid_int:03d} {name} ({types_str})" if isinstance(pid_int, int) else f"{name} ({types_str})"
        results.append((pid_int or 0, line))

    # Dedupe while keeping smallest id for stable ordering.
    best: dict[str, int] = {}
    formatted: dict[str, str] = {}
    for pid_int, line in results:
        if line not in best or pid_int < best[line]:
            best[line] = pid_int
            formatted[line] = line

    lines = sorted(formatted.values(), key=lambda s: best[s])
    if not lines:
        return f"未找到匹配 '{q}' 的宝可梦"
    return f"**搜索结果 ({len(lines)})**:\n" + "\n".join(lines)


@tool(args_schema=PokemonNameSchema)
def get_evolution_chain(pokemon_name: str) -> str:
    """获取宝可梦的进化链"""
    q = (pokemon_name or "").strip()
    if not q:
        return "请提供宝可梦名称"

    data = get_pokemon_data()
    resolved = data.resolve_name(q) or q
    rec = data.get_by_cn_name(resolved)
    if not rec:
        return f"未找到宝可梦: {q}"

    chain = evolution_chain(resolved, data=data)
    if len(chain) <= 1:
        return f"{resolved} 没有进化链信息(可能是无法进化或数据缺失)"

    index = chain.index(resolved) if resolved in chain else 0
    chain_display = " → ".join([f"**{p}**" if p == resolved else p for p in chain])
    return f"**{resolved}** 的进化链:\n{chain_display}\n\n当前为第 {index + 1}/{len(chain)} 阶段"


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
    """宝可梦图鉴查询代理 - 搜索宝可梦、进化链、招式、特性"""

    def __init__(self):
        tools = [search_pokedex, get_evolution_chain, get_pokemon_moves, get_ability_info]
        super().__init__(tools=tools, bind_tools=True)
        logger.info("PokedexAgent initialized")

    def get_info(self) -> dict:
        return {
            "name": "PokedexAgent",
            "description": "宝可梦图鉴查询代理 - 搜索宝可梦、进化链、招式、特性",
            "tools": [t.name for t in self._tools],
        }

    # 使用 ToolAgent 基类的默认实现:
    # - _build_graph()
    # - _call_model()
    # - _should_continue()
    # - _run_tool()


if __name__ == "__main__":
    print(search_pokedex.invoke({"query": "电"}))
    print(get_evolution_chain.invoke({"pokemon_name": "皮卡丘"}))
    print(get_pokemon_moves.invoke({"pokemon_name": "喷火龙"}))
    print(get_ability_info.invoke({"ability_name": "静电"}))
