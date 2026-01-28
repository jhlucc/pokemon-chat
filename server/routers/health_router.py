from __future__ import annotations

import os

from fastapi import APIRouter

from src.core.feature_flags import feature_enabled
from src.core.settings import settings
from src.utils.net import parse_host_port, tcp_check

health = APIRouter(tags=["health"])


@health.get("/healthz")
async def healthz():
    return {"status": "ok"}


@health.get("/readyz")
async def readyz():
    checks: dict = {}
    warnings: list[str] = []

    # Lightweight config warnings (no network calls).
    from src.core.provider_config import get_provider_api_key

    if not (
        settings.llm.api_key
        or get_provider_api_key("siliconflow")
        or get_provider_api_key("openai")
        or get_provider_api_key("deepseek")
        or get_provider_api_key("zhipu")
    ):
        warnings.append("No LLM API key configured (llm_api_key / SILICONFLOW_API_KEY / OPENAI_API_KEY).")
    if feature_enabled("enable_web_search") and not (settings.tavily.api_key):
        warnings.append("Web search is enabled but tavily_api_key is empty.")

    # Neo4j (bolt)
    neo4j_enabled = bool(feature_enabled("enable_knowledge_graph"))
    neo4j_host, neo4j_port = parse_host_port(settings.database.neo4j_uri, default_port=7687)
    neo4j_ok, neo4j_err = tcp_check(neo4j_host, neo4j_port) if neo4j_enabled else (True, "")
    checks["neo4j"] = {
        "enabled": neo4j_enabled,
        "target": f"{neo4j_host}:{neo4j_port}",
        "ok": neo4j_ok,
        "error": neo4j_err,
    }

    # Milvus (best-effort TCP check on configured URI)
    kb_enabled = bool(feature_enabled("enable_knowledge_base"))
    milvus_host, milvus_port = parse_host_port(settings.database.milvus_uri, default_port=19530)
    milvus_ok, milvus_err = tcp_check(milvus_host, milvus_port) if kb_enabled else (True, "")
    checks["milvus"] = {
        "enabled": kb_enabled,
        "target": f"{milvus_host}:{milvus_port}",
        "ok": milvus_ok,
        "error": milvus_err,
    }

    # MySQL
    mysql_enabled = bool(feature_enabled("enable_mcp"))  # MCP tool uses MySQL in this project
    mysql_ok, mysql_err = (
        tcp_check(settings.database.mysql_host, settings.database.mysql_port) if mysql_enabled else (True, "")
    )
    checks["mysql"] = {
        "enabled": mysql_enabled,
        "target": f"{settings.database.mysql_host}:{settings.database.mysql_port}",
        "ok": mysql_ok,
        "error": mysql_err,
    }

    # FunASR (ASR)
    funasr_enabled = bool(feature_enabled("enable_asr"))
    funasr_host, funasr_port = parse_host_port(settings.asr.funasr_url, default_port=10095)
    funasr_ok, funasr_err = tcp_check(funasr_host, funasr_port) if funasr_enabled else (True, "")
    checks["funasr"] = {
        "enabled": funasr_enabled,
        "target": f"{funasr_host}:{funasr_port}",
        "ok": funasr_ok,
        "error": funasr_err,
    }

    ok = all(v["ok"] for v in checks.values() if v.get("enabled"))
    return {
        "status": "ok" if ok else "fail",
        "app": {
            "version": (os.getenv("APP_VERSION") or "").strip() or None,
            "build_sha": (os.getenv("BUILD_SHA") or "").strip() or None,
            "build_time": (os.getenv("BUILD_TIME") or "").strip() or None,
        },
        "features": {
            "enable_knowledge_base": bool(feature_enabled("enable_knowledge_base")),
            "enable_knowledge_graph": bool(feature_enabled("enable_knowledge_graph")),
            "enable_web_search": bool(feature_enabled("enable_web_search")),
            "enable_mcp": bool(feature_enabled("enable_mcp")),
            "enable_reranker": bool(feature_enabled("enable_reranker")),
            "enable_asr": bool(feature_enabled("enable_asr")),
            "enable_ner_bert": bool(feature_enabled("enable_ner_bert")),
        },
        "checks": checks,
        "warnings": warnings,
    }
