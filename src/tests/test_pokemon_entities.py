from __future__ import annotations


def test_extract_entities_cn():
    from src.agents.pokemon_entities import extract_pokemon_entities

    assert extract_pokemon_entities("皮卡丘的属性是什么？") == ["皮卡丘"]


def test_extract_entities_en_case_insensitive():
    from src.agents.pokemon_entities import extract_pokemon_entities

    assert extract_pokemon_entities("Pikachu ability?") == ["皮卡丘"]
