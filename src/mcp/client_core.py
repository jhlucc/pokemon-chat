from __future__ import annotations

import json
import os
from contextlib import AsyncExitStack
from typing import Optional, Tuple

from mcp import ClientSession
from mcp.client.sse import sse_client
from openai import OpenAI

from src.core.settings import settings
from src.utils.http_client import get_safe_httpx_client


class MCPClient:
    """
    Minimal MCP client:
    - Connects to an MCP server via SSE
    - Uses an OpenAI-compatible chat completion model to decide tool calls

    No hardcoded API keys.
    """

    def __init__(
        self,
        sse_url: Optional[str] = None,
        *,
        llm_api_key: Optional[str] = None,
        llm_base_url: Optional[str] = None,
        llm_model: Optional[str] = None,
    ) -> None:
        self.sse_url = sse_url or os.getenv("MCP_SSE_URL") or os.getenv("mcp_sse_url") or "http://127.0.0.1:8000/sse"

        # Allow dedicated MCP LLM config; fall back to global LLM settings.
        self.llm_model = (
            llm_model
            or os.getenv("MCP_LLM_MODEL")
            or os.getenv("mcp_llm_model")
            or settings.llm.model_name
            or "deepseek-chat"
        )
        self.llm_api_key = (
            llm_api_key
            or os.getenv("MCP_LLM_API_KEY")
            or os.getenv("mcp_llm_api_key")
            or os.getenv("DEEPSEEK_API_KEY")
            or (settings.llm.api_key if settings.features.enable_mcp else None)
        )
        self.llm_base_url = (
            llm_base_url
            or os.getenv("MCP_LLM_BASE_URL")
            or os.getenv("mcp_llm_base_url")
            or os.getenv("DEEPSEEK_BASE_URL")
            or settings.llm.api_base
            or "https://api.deepseek.com"
        )

        # OpenAI client is cheap; if the key is missing we delay the error to ask().
        self._client: Optional[OpenAI] = None
        if self.llm_api_key:
            self._client = OpenAI(
                api_key=self.llm_api_key,
                base_url=self.llm_base_url,
                http_client=get_safe_httpx_client(),
            )

    async def aclose(self) -> None:
        # Kept for router compatibility; OpenAI client doesn't need explicit close.
        return

    async def ask(self, query: str) -> Tuple[str, Optional[str]]:
        if self._client is None:
            raise ValueError(
                "MCPClient missing API key. Set MCP_LLM_API_KEY (or DEEPSEEK_API_KEY / llm_api_key)."
            )

        async with AsyncExitStack() as stack:
            rd, wr = await stack.enter_async_context(sse_client(self.sse_url))
            session = await stack.enter_async_context(ClientSession(read_stream=rd, write_stream=wr))
            await session.initialize()

            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.inputSchema,
                    },
                }
                for t in (await session.list_tools()).tools
            ]

            msgs = [{"role": "user", "content": query}]
            first = self._client.chat.completions.create(
                model=self.llm_model, messages=msgs, tools=tools, tool_choice="auto"
            )
            choice = first.choices[0]

            coords_json: Optional[str] = None
            if choice.finish_reason == "tool_calls":
                msgs.append(choice.message.model_dump())
                for tc in choice.message.tool_calls:
                    args = json.loads(tc.function.arguments)
                    res = await session.call_tool(tc.function.name, args)
                    coords_json = res.content[0].text if res.content else None
                    msgs.append(
                        {
                            "role": "tool",
                            "name": tc.function.name,
                            "tool_call_id": tc.id,
                            "content": coords_json or "",
                        }
                    )
                final = self._client.chat.completions.create(model=self.llm_model, messages=msgs)
                answer = final.choices[0].message.content
            else:
                answer = choice.message.content

            return answer, coords_json


# ──────────────────── 单例 + 缓存 ────────────────────
from functools import lru_cache
from typing import Dict
import time

_mcp_client: Optional[MCPClient] = None
_query_cache: Dict[str, tuple] = {}  # {query: (answer, coords, timestamp)}
_cache_ttl = 300  # 5 minutes

def get_mcp_client() -> MCPClient:
    """获取 MCP 客户端单例"""
    global _mcp_client
    if _mcp_client is None:
        _mcp_client = MCPClient()
    return _mcp_client

async def cached_ask(query: str, use_cache: bool = True) -> Tuple[str, Optional[str]]:
    """带缓存的 MCP 查询"""
    global _query_cache
    
    # 检查缓存
    if use_cache and query in _query_cache:
        answer, coords, ts = _query_cache[query]
        if time.time() - ts < _cache_ttl:
            return answer, coords
        else:
            del _query_cache[query]
    
    # 执行查询
    client = get_mcp_client()
    answer, coords = await client.ask(query)
    
    # 存入缓存
    if use_cache and answer:
        _query_cache[query] = (answer, coords, time.time())
    
    return answer, coords


if __name__ == "__main__":
    import asyncio

    async def _test():
        client = MCPClient()
        try:
            answer, coords = await client.ask("皮卡丘出现在真实世界的具体坐标？")
            print("=== ANSWER ===")
            print(answer)
            print("\n=== COORDS_JSON ===")
            print(coords)
        finally:
            await client.aclose()

    asyncio.run(_test())
