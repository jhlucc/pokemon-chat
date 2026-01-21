import os

# Ensure offline-safe defaults for this test module.
os.environ.setdefault("enable_reranker", "false")
os.environ.setdefault("enable_web_search", "false")
os.environ.setdefault("enable_knowledge_base", "false")
os.environ.setdefault("enable_knowledge_graph", "false")
os.environ.setdefault("enable_mcp", "false")

# Clear provider keys so model selection behaves deterministically.
for _k in (
    "SILICONFLOW_API_KEY",
    "OPENAI_API_KEY",
    "DEEPSEEK_API_KEY",
    "MCP_LLM_API_KEY",
    "mcp_llm_api_key",
):
    os.environ.pop(_k, None)

import asyncio
import unittest

from src.mcp.client_core import MCPClient
from src.models import select_model
from src.runtime import get_kb, reset_all


class OfflineSmokeTests(unittest.TestCase):
    def tearDown(self):
        # Avoid cross-test pollution from cached singletons.
        reset_all()

    def test_select_model_missing_key_raises(self):
        with self.assertRaises(ValueError):
            select_model(model_provider="siliconflow", model_name="Qwen/Qwen2.5-7B-Instruct")

    def test_knowledge_base_disabled_raises_on_use(self):
        kb = get_kb()
        with self.assertRaises(RuntimeError):
            kb.search(query="hi", db_id="kb_dummy")

    def test_mcp_client_missing_key_raises(self):
        async def _run():
            client = MCPClient()
            with self.assertRaises(ValueError):
                await client.ask("hi")

        asyncio.run(_run())

