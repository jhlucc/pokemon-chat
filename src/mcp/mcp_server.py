"""
FastMCP 版本的 Pokémon-MySQL Server
----------------------------------
✔ 两个工具：
    1. search_locations_by_pokemon
    2. get_location_info
"""

import os, json, logging
from mysql.connector import connect, Error
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from mcp.types import TextContent         # 仅用来包装返回值
from typing import List

# ──────────────────── 环境 & 日志 ────────────────────
load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

DB_CONF = dict(
    host=os.getenv("MYSQL_HOST", "127.0.0.1"),
    user=os.getenv("MYSQL_USER","root"),
    password=os.getenv("MYSQL_PASSWORD","123456"),
    database="langgraph",
    port = 3307,
    charset="utf8mb4"
)

app = FastMCP("pokemon-fastmcp")

ERR_TXT = lambda msg: [TextContent(type="text",
                                   text=json.dumps({"error": msg}, ensure_ascii=False))]


# ──────────────────── 公共辅助 ────────────────────
def _rows_to_text(rows) -> List[TextContent]:
    if not rows:
        return ERR_TXT("未找到匹配地点")
    data = [{"location": name, "lat": lat, "lng": lng} for lat, lng, name in rows]
    return [TextContent(type="text",
                        text=json.dumps(data, ensure_ascii=False))]
# ──────────────────── 工具 1 ────────────────────
@app.tool(
    name="search_locations_by_pokemon",
    description="模糊搜索宝可梦出现地点，返回经纬度与真实地名 JSON 列表"
)
def search_locations_by_pokemon(pokemon_name: str) -> List[TextContent]:
    sql = """
        SELECT latitude, longitude, real_location
          FROM pokemon_locations
         WHERE pokemon_list LIKE %s
    """
    pattern = f"%{pokemon_name}%"
    try:
        with connect(**DB_CONF) as conn, conn.cursor() as cur:
            cur.execute(sql, (pattern,))
            rows = cur.fetchall()
    except Error as e:
        logger.error("DB error: %s", e)
        return ERR_TXT(f"数据库错误：{e}")

    return _rows_to_text(rows)

# ──────────────────── 工具 2 ────────────────────
@app.tool(
    name="get_location_info",
    description="输入地点（宝可梦世界名或现实地名）→ 返回匹配的经纬度与真实地名 JSON 列表"
)
def get_location_info(location: str) -> List[TextContent]:
    sql = """
        SELECT latitude, longitude, real_location
          FROM pokemon_locations
         WHERE pokemon_region = %s OR real_location = %s
    """
    try:
        with connect(**DB_CONF) as conn, conn.cursor() as cur:
            cur.execute(sql, (location, location))
            rows = cur.fetchall()
    except Error as e:
        logger.error("DB error: %s", e)
        return ERR_TXT(f"数据库错误：{e}")

    return _rows_to_text(rows)

# ──────────────────── 运行 ────────────────────
if __name__ == "__main__":
    app.run(transport="sse")
