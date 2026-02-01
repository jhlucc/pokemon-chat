from __future__ import annotations

from langchain_core.messages import HumanMessage

from src.graph.nodes.graph_worker import GraphWorker


def test_graph_worker_evolution_query_uses_local_dataset_and_hides_cypher():
    worker = GraphWorker()
    out = worker(
        {
            "messages": [
                HumanMessage(
                    content=(
                        '喷火龙进化是什么 MATCH (p:Pokémon {name: "喷火龙"})-[:evolves_into]->(e:Pokémon) RETURN e.name'
                    )
                )
            ]
        }
    )

    msg = out["messages"][-1]
    content = getattr(msg, "content", "")

    # Should answer with the known local evolution chain.
    assert "小火龙" in content
    assert "火恐龙" in content
    assert "喷火龙" in content

    # Should not leak Cypher into the user-facing response.
    assert "MATCH" not in content
    assert "RETURN" not in content
