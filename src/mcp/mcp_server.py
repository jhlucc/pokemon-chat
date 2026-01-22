"""
FastMCP 版本的 Pokémon-MySQL Server
----------------------------------
✔ 两个工具：
    1. search_locations_by_pokemon
    2. get_location_info
"""

from __future__ import annotations

import json
import logging
import os
from typing import List, Sequence, Tuple

import pymysql
from mcp.server.fastmcp import FastMCP
from mcp.types import TextContent  # 仅用来包装返回值

# ──────────────────── 日志 ────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

def _get_db_conf() -> dict:
    # Keep compatibility with both legacy UPPERCASE env names and the project's settings keys.
    host = os.getenv("MYSQL_HOST") or os.getenv("mysql_host") or "127.0.0.1"
    user = os.getenv("MYSQL_USER") or os.getenv("mysql_user") or "root"
    password = os.getenv("MYSQL_PASSWORD") or os.getenv("mysql_password") or ""
    database = os.getenv("MYSQL_DATABASE") or os.getenv("mysql_database") or "langgraph"
    port = int(os.getenv("MYSQL_PORT") or os.getenv("mysql_port") or "3306")
    return dict(
        host=host,
        user=user,
        password=password,
        database=database,
        port=port,
        charset="utf8mb4",
        cursorclass=pymysql.cursors.Cursor,
    )

# ──────────────────── 连接池 ────────────────────
_pool = None

def _get_connection():
    """获取连接池中的连接"""
    global _pool
    if _pool is None:
        try:
            from dbutils.pooled_db import PooledDB
            _pool = PooledDB(
                creator=pymysql,
                maxconnections=10,
                mincached=2,
                maxcached=5,
                blocking=True,
                **_get_db_conf()
            )
            logger.info("MySQL connection pool initialized")
        except ImportError:
            logger.warning("DBUtils not installed, falling back to direct connections")
            return pymysql.connect(**_get_db_conf())
    return _pool.connection()

app = FastMCP(
    "pokemon-fastmcp",
    host=os.getenv("MCP_HOST", "0.0.0.0"),
    port=int(os.getenv("MCP_PORT", "8000")),
)

def _err_txt(msg: str) -> List[TextContent]:
    return [TextContent(type="text", text=json.dumps({"error": msg}, ensure_ascii=False))]


# ──────────────────── 公共辅助 ────────────────────
def _rows_to_text(rows: Sequence[Tuple[float, float, str]]) -> List[TextContent]:
    if not rows:
        return _err_txt("未找到匹配地点")
    data = [{"location": name, "lat": float(lat), "lng": float(lng)} for lat, lng, name in rows]
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
        conn = _get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(sql, (pattern,))
                rows = cur.fetchall()
        finally:
            conn.close()
    except Exception as e:
        logger.error("DB error: %s", e)
        return _err_txt(f"数据库错误：{e}")

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
        conn = _get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(sql, (location, location))
                rows = cur.fetchall()
        finally:
            conn.close()
    except Exception as e:
        logger.error("DB error: %s", e)
        return _err_txt(f"数据库错误：{e}")

    return _rows_to_text(rows)

# ──────────────────── 运行 ────────────────────
if __name__ == "__main__":
    app.run(transport="sse")
