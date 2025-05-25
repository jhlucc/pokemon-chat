# client_core.py
import os, json
from contextlib import AsyncExitStack
from openai import OpenAI
from mcp.client.sse import sse_client
from mcp import ClientSession

DEESEEK_KEY = os.getenv("DEEPSEEK_API_KEY", "sk-b0f9874b7bac423bb9be095b4157e426")

class MCPClient:
    def __init__(self, sse_url: str = "http://127.0.0.1:8000/sse"):
        self.sse_url = sse_url
        self.openai  = OpenAI(api_key=DEESEEK_KEY,
                              base_url="https://api.deepseek.com")

    async def ask(self, query: str):
        async with AsyncExitStack() as stack:
            # ── 1. 建立 SSE / Session ───────────────────
            rd, wr = await stack.enter_async_context(sse_client(self.sse_url))
            session = await stack.enter_async_context(
                ClientSession(read_stream=rd, write_stream=wr))
            await session.initialize()

            # ── 2. 拉工具列表 ───────────────────────────
            tools = [{"type": "function",
                      "function": {"name": t.name,
                                   "description": t.description,
                                   "parameters": t.inputSchema}}
                     for t in (await session.list_tools()).tools]

            # ── 3. 第一轮对话 ───────────────────────────
            msgs = [{"role": "user", "content": query}]
            first = self.openai.chat.completions.create(
                model="deepseek-chat", messages=msgs,
                tools=tools, tool_choice="auto")
            choice = first.choices[0]

            coords_json = None
            # ── 4. 可能执行工具 ─────────────────────────
            if choice.finish_reason == "tool_calls":
                msgs.append(choice.message.model_dump())
                for tc in choice.message.tool_calls:
                    args = json.loads(tc.function.arguments)
                    res  = await session.call_tool(tc.function.name, args)
                    coords_json = res.content[0].text
                    msgs.append({"role": "tool",
                                 "name": tc.function.name,
                                 "tool_call_id": tc.id,
                                 "content": coords_json})
                # ── 5. 最终回复 ────────────────────────
                final = self.openai.chat.completions.create(
                    model="deepseek-chat", messages=msgs)
                answer = final.choices[0].message.content
            else:
                answer = choice.message.content

            return answer, coords_json

if __name__ == "__main__":
    import asyncio
    async def _test():
        client = MCPClient()                     # 默认连 127.0.0.1:8000/sse
        try:
            answer, coords = await client.ask("皮卡丘出现在真实世界的具体坐标")
            print("=== ANSWER ===")
            print(answer)
            print("\n=== COORDS_JSON ===")
            print(coords)
        finally:
            await client.aclose()

    asyncio.run(_test())
# 根据现有数据，以下是皮卡丘在真实世界中可能出现的具体坐标（部分地点无精确经纬度）：
#
# 1. **日本町田市**
#    📍 坐标：35.546656°N, 139.4385568°E
#    🏙️ 东京都附近的卫星城市，曾出现在《精灵宝可梦》动画场景中。
#
# 2. **美国曼哈顿下城**
#    📍 坐标：40.7208595°N, -74.0006686°W
#    🗽 纽约市核心区域，皮卡丘曾在此参与现实联动活动（如2019年「Pokémon GO Fest」）。
#
# 3. **美国檀香山**
#    📍 坐标：21.3098845°N, -157.8581401°W
#    🌴 夏威夷首府，官方活动或电影取景地（如《大侦探皮卡丘》宣传）。
#
# 4. **法国圣卢**
#    📍 坐标：46.135685°N, 2.272921°E
#    🏰 中部乡村地区，可能与宝可梦主题旅行活动相关。
#
# 其他未标注坐标的地点（如秩父山地、东海发电所等）多为动画或游戏中的虚构场景原型，暂无公开地理数据。如需实时活动定位，建议参考《Pokémon GO》官方活动公告或AR游戏中的动态出现点。
#
# === COORDS_JSON ===
# [{"location": "町田市", "lat": 35.546656, "lng": 139.4385568}, {"location": "秩父山地", "lat": null, "lng": null}, {"location": "东海发电所", "lat": null, "lng": null}, {"location": "曼哈顿下城", "lat": 40.7208595, "lng": -74.0006686}, {"location": "檀香山", "lat": 21.3098845, "lng": -157.8581401}, {"location": "哲比", "lat": null, "lng": null}, {"location": "圣卢", "lat": 46.135685, "lng": 2.272921}, {"location": "道格拉斯", "lat": null, "lng": null}, {"location": "小马恩海峡", "lat": null, "lng": null}, {"location": "鹿屋市", "lat": 31.3782477, "lng": 130.8522618}]
