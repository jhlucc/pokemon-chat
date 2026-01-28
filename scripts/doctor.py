from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure repo root is on sys.path even when running `python scripts/doctor.py`.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.core.feature_flags import feature_enabled  # noqa: E402
from src.core.settings import settings  # noqa: E402
from src.utils.net import parse_host_port, tcp_check  # noqa: E402


def _mask(name: str, value: str) -> str:
    if not value:
        return f"{name}: missing"
    return f"{name}: set (len={len(value)})"


def main() -> int:
    parser = argparse.ArgumentParser(description="Project sanity checks (env + optional service reachability).")
    parser.add_argument("--timeout", type=float, default=0.5, help="TCP connect timeout in seconds (default: 0.5)")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if any enabled dependency check fails or required keys are missing.",
    )
    args = parser.parse_args()

    print(f"Python: {sys.version.split()[0]}")
    print(f"Repo:   {settings.paths.base_dir}")
    print()

    print("Features:")
    for k in [
        "enable_knowledge_base",
        "enable_knowledge_graph",
        "enable_web_search",
        "enable_mcp",
        "enable_reranker",
        "enable_asr",
        "enable_ner_bert",
    ]:
        print(f"  {k} = {bool(feature_enabled(k))}")
    print()

    print("Keys:")
    print(f"  {_mask('llm_api_key', settings.llm.api_key)}")
    print(f"  {_mask('tavily_api_key', settings.tavily.api_key)}")
    print(f"  {_mask('tool_openweather_api_key', settings.tools.openweather_api_key)}")
    print()

    problems: list[str] = []

    if not settings.llm.api_key:
        problems.append("llm_api_key is empty")
    if feature_enabled("enable_web_search") and not settings.tavily.api_key:
        problems.append("enable_web_search=true but tavily_api_key is empty")

    print("Reachability:")
    neo4j_enabled = bool(feature_enabled("enable_knowledge_graph"))
    kb_enabled = bool(feature_enabled("enable_knowledge_base"))
    mysql_enabled = bool(feature_enabled("enable_mcp"))
    funasr_enabled = bool(feature_enabled("enable_asr"))

    neo4j_host, neo4j_port = parse_host_port(settings.database.neo4j_uri, default_port=7687)
    milvus_host, milvus_port = parse_host_port(settings.database.milvus_uri, default_port=19530)
    funasr_host, funasr_port = parse_host_port(settings.asr.funasr_url, default_port=10095)

    checks: list[tuple[str, bool, str, int]] = [
        ("neo4j", neo4j_enabled, neo4j_host, neo4j_port),
        ("milvus", kb_enabled, milvus_host, milvus_port),
        ("mysql", mysql_enabled, settings.database.mysql_host, settings.database.mysql_port),
        ("funasr", funasr_enabled, funasr_host, funasr_port),
    ]

    for name, enabled, host, port in checks:
        if not enabled:
            print(f"  {name}: skipped (disabled)")
            continue
        ok, err = tcp_check(host, port, timeout_s=args.timeout)
        if ok:
            print(f"  {name}: ok ({host}:{port})")
        else:
            print(f"  {name}: fail ({host}:{port}) - {err}")
            problems.append(f"{name} not reachable: {host}:{port}")

    if problems:
        print()
        print("Problems:")
        for p in problems:
            print(f"  - {p}")

    if args.strict and problems:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
