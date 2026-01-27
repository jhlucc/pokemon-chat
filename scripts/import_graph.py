"""
Import Neo4j graph data from JSON files under `resources/data/kg_data`.

This is a local bootstrap script (dev/ops utility), not a runtime dependency.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

from py2neo import Graph, Node


def _ensure_utf8_stdout() -> None:
    # Best-effort: mainly for Windows consoles where UTF-8 isn't the default.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


class MedicalGraphFromJson:
    def __init__(self) -> None:
        cur_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.data_path = os.path.join(cur_dir, "resources", "data", "kg_data")

        neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        neo4j_user = os.getenv("NEO4J_USERNAME", "neo4j")
        neo4j_password = os.getenv("NEO4J_PASSWORD", "")

        self.g = Graph(neo4j_uri, auth=(neo4j_user, neo4j_password))
        self.rel_file = "relations.json"
        self.node_file = "entities.json"

    def build_graph(self) -> None:
        res = self.build_nodes()
        if res == -1:
            print("no nodes file, can not create relations")
            return
        self.build_rels()

    def build_nodes(self) -> int:
        node_file = os.path.join(self.data_path, self.node_file)
        if not os.path.exists(node_file):
            return -1
        with open(node_file, encoding="utf-8") as f:
            nodes: list[dict[str, Any]] = json.load(f)
        for node in nodes:
            self.create_node(node)
        return 0

    def create_node(self, node: dict[str, Any]) -> None:
        label = node["label"]
        payload = node["name"]

        if label == "Person":
            n = Node(
                label,
                name=payload.get("chinese_name", ""),
                japanese_name=payload.get("japanese_name", ""),
                english_name=payload.get("english_name", ""),
                gender=payload.get("gender", ""),
            )
        elif label in {"Pokemon", "Pokémon"}:
            # Keep attribute bag as JSON string for compatibility with existing schemas.
            n = Node(
                label,
                name=payload.get("chinese_name", ""),
                japanese_name=payload.get("japanese_name", ""),
                english_name=payload.get("english_name", ""),
                ability=payload.get("ability", ""),
                hidden_ability=payload.get("hidden_ability", ""),
                height=payload.get("height", ""),
                weight=payload.get("weight", ""),
                evolution_level=payload.get("evolution_level", ""),
                attr_ability=json.dumps(payload.get("attr_ability", {}), ensure_ascii=False),
            )
        else:
            n = Node(label, name=payload)

        self.g.create(n)

    def build_rels(self) -> None:
        rel_file = os.path.join(self.data_path, self.rel_file)
        if not os.path.exists(rel_file):
            print(f"{self.rel_file} not exist, skip")
            return

        with open(rel_file, encoding="utf-8") as f:
            relations: list[dict[str, Any]] = json.load(f)

        for rel in relations:
            self.create_rel(rel)

    def create_rel(self, rels_set: dict[str, Any]) -> None:
        cnt = 0
        start_entity_type = rels_set["start_entity_type"]
        end_entity_type = rels_set["end_entity_type"]
        rel_type = rels_set["rel_type"]
        rel_name = rels_set["rel_name"]
        rels = rels_set["rels"]

        # Labels / relationship types can't be parametrized in Cypher; keep them interpolated,
        # but pass property values as parameters.
        query = (
            f"MATCH (p:{start_entity_type}), (q:{end_entity_type}) "
            "WHERE p.name=$p_name AND q.name=$q_name "
            f"CREATE (p)-[rel:{rel_type}{{name:$rel_name}}]->(q)"
        )

        for rel in rels:
            p_name = rel["start_entity_name"]
            q_name = rel["end_entity_name"]
            try:
                self.g.run(query, p_name=p_name, q_name=q_name, rel_name=rel_name)
                cnt += 1
                print(f"{rel_type} {cnt}/{len(rels)}")
            except Exception as e:
                print(e)


if __name__ == "__main__":
    _ensure_utf8_stdout()
    handler = MedicalGraphFromJson()
    handler.build_graph()
