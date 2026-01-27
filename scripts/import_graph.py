"""
Import Neo4j knowledge-graph data from JSON files under `resources/data/kg_data`.

Designed to work both locally and in Docker Compose:
- waits for Neo4j to be ready (optional)
- idempotent bootstrap (skips if a marker node exists)
- optional reset / force re-import
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from py2neo import Graph


def _ensure_utf8_stdout() -> None:
    # Best-effort: mainly for Windows consoles where UTF-8 isn't the default.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _env(*names: str, default: str = "") -> str:
    for name in names:
        value = os.getenv(name)
        if value is not None and str(value).strip() != "":
            return str(value)
    return default


def _cypher_ident(name: str) -> str:
    # Always backtick-quote labels/types to handle non-ascii like `Pokémon`.
    return f"`{name.replace('`', '``')}`"


def _connect_with_retry(uri: str, username: str, password: str, wait_seconds: int) -> Graph:
    deadline = time.time() + max(0, wait_seconds)
    last_err: Exception | None = None

    while True:
        try:
            # docker-compose.yml may use `NEO4J_AUTH=none` (no auth). In that case, avoid sending auth.
            graph = Graph(uri, auth=(username, password)) if password else Graph(uri)
            graph.run("RETURN 1").evaluate()
            return graph
        except Exception as e:  # noqa: BLE001
            last_err = e
            if time.time() >= deadline:
                break
            print(f"[neo4j] waiting... ({e})")
            time.sleep(2)

    raise RuntimeError(f"Failed to connect to Neo4j at {uri} within {wait_seconds}s: {last_err}") from last_err


def _load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Missing data file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _bootstrap_marker_exists(graph: Graph, marker_id: str) -> bool:
    q = "MATCH (n:__PokemonChatBootstrap {id:$id}) RETURN count(n) AS c"
    return int(graph.run(q, id=marker_id).evaluate() or 0) > 0


def _write_bootstrap_marker(graph: Graph, marker_id: str, source: str) -> None:
    q = """
MERGE (n:__PokemonChatBootstrap {id:$id})
SET n.source=$source, n.updated_at=timestamp()
"""
    graph.run(q, id=marker_id, source=source)


def _reset_graph(graph: Graph) -> None:
    graph.run("MATCH (n) DETACH DELETE n")


def import_nodes(graph: Graph, nodes: list[dict[str, Any]], batch_size: int = 500) -> None:
    tx = graph.begin()
    for idx, node in enumerate(nodes, 1):
        label = str(node["label"])
        payload = node["name"]

        if label == "Person":
            params = {
                "name": payload.get("chinese_name", ""),
                "japanese_name": payload.get("japanese_name", ""),
                "english_name": payload.get("english_name", ""),
                "gender": payload.get("gender", ""),
            }
            query = f"""
MERGE (n:{_cypher_ident(label)} {{name:$name}})
SET n.japanese_name=$japanese_name,
    n.english_name=$english_name,
    n.gender=$gender
"""
        elif label in {"Pokemon", "Pokémon"}:
            params = {
                "name": payload.get("chinese_name", ""),
                "japanese_name": payload.get("japanese_name", ""),
                "english_name": payload.get("english_name", ""),
                "ability": payload.get("ability", ""),
                "hidden_ability": payload.get("hidden_ability", ""),
                "height": payload.get("height", ""),
                "weight": payload.get("weight", ""),
                "evolution_level": payload.get("evolution_level", ""),
                # Keep attribute bag as JSON string for compatibility with existing schemas.
                "attr_ability": json.dumps(payload.get("attr_ability", {}), ensure_ascii=False),
            }
            query = f"""
MERGE (n:{_cypher_ident(label)} {{name:$name}})
SET n.japanese_name=$japanese_name,
    n.english_name=$english_name,
    n.ability=$ability,
    n.hidden_ability=$hidden_ability,
    n.height=$height,
    n.weight=$weight,
    n.evolution_level=$evolution_level,
    n.attr_ability=$attr_ability
"""
        else:
            params = {"name": str(payload)}
            query = f"MERGE (n:{_cypher_ident(label)} {{name:$name}})"

        tx.run(query, **params)
        if idx % batch_size == 0:
            graph.commit(tx)
            tx = graph.begin()
            print(f"[nodes] {idx}/{len(nodes)}")

    graph.commit(tx)
    print(f"[nodes] {len(nodes)}/{len(nodes)}")


def import_relationships(graph: Graph, rel_groups: list[dict[str, Any]]) -> None:
    for group in rel_groups:
        start_label = _cypher_ident(str(group["start_entity_type"]))
        end_label = _cypher_ident(str(group["end_entity_type"]))
        rel_type = _cypher_ident(str(group["rel_type"]))
        rel_name = str(group["rel_name"])
        rels = group["rels"] or []

        # Labels / relationship types can't be parametrized in Cypher; keep them interpolated.
        query = f"""
UNWIND $rows AS row
MATCH (p:{start_label} {{name: row.p_name}}), (q:{end_label} {{name: row.q_name}})
MERGE (p)-[rel:{rel_type} {{name: $rel_name}}]->(q)
"""
        rows = [{"p_name": r["start_entity_name"], "q_name": r["end_entity_name"]} for r in rels]
        graph.run(query, rows=rows, rel_name=rel_name)
        print(f"[rels] {str(group['rel_type'])}: {len(rows)}")


def main() -> int:
    _ensure_utf8_stdout()

    parser = argparse.ArgumentParser(description="Import Pokemon graph data into Neo4j.")
    parser.add_argument(
        "--data-dir",
        default=str(_repo_root() / "resources" / "data" / "kg_data"),
        help="Directory containing entities.json + relations.json",
    )
    parser.add_argument("--wait-seconds", type=int, default=int(_env("NEO4J_WAIT_SECONDS", default="60") or 60))
    parser.add_argument("--marker-id", default=_env("KG_BOOTSTRAP_MARKER_ID", default="pokemon-chat-kg-v1"))
    parser.add_argument("--force", action="store_true", help="Force import even if marker exists.")
    parser.add_argument("--reset", action="store_true", help="DANGEROUS: delete all nodes before importing.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    entities_path = data_dir / "entities.json"
    relations_path = data_dir / "relations.json"

    uri = _env("NEO4J_URI", "neo4j_uri", default="bolt://localhost:7687")
    username = _env("NEO4J_USERNAME", "neo4j_username", default="neo4j")
    password = _env("NEO4J_PASSWORD", "neo4j_password", default="")

    print(f"[import_graph] data_dir={data_dir}")
    print(f"[import_graph] neo4j={uri} user={username} auth={'on' if password else 'off'}")

    graph = _connect_with_retry(uri, username, password, wait_seconds=args.wait_seconds)

    if not args.force and _bootstrap_marker_exists(graph, args.marker_id):
        print(f"[import_graph] marker exists ({args.marker_id}), skip.")
        return 0

    if args.reset:
        print("[import_graph] resetting database (DETACH DELETE)...")
        _reset_graph(graph)

    nodes: list[dict[str, Any]] = _load_json(entities_path)
    rel_groups: list[dict[str, Any]] = _load_json(relations_path)

    print(f"[import_graph] importing nodes: {len(nodes)}")
    import_nodes(graph, nodes)
    print(f"[import_graph] importing relationships groups: {len(rel_groups)}")
    import_relationships(graph, rel_groups)

    _write_bootstrap_marker(
        graph,
        args.marker_id,
        source=f"kg_data:{entities_path.name},{relations_path.name}",
    )

    print("[import_graph] done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
