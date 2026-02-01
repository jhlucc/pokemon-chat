from __future__ import annotations


def test_format_basic_facts_includes_type():
    from src.agents.pokemon_data import get_pokemon_data
    from src.agents.pokemon_facts import format_basic_facts

    data = get_pokemon_data()
    pikachu = data.get_by_cn_name("皮卡丘")
    assert pikachu is not None

    text = format_basic_facts(pikachu)
    assert "属性" in text
    assert "电" in text


def test_format_evolution_chain_contains_pre_and_next():
    from src.agents.pokemon_data import get_pokemon_data
    from src.agents.pokemon_facts import format_evolution

    data = get_pokemon_data()
    pikachu = data.get_by_cn_name("皮卡丘")
    assert pikachu is not None

    text = format_evolution(pikachu)
    assert "皮丘" in text
    assert "皮卡丘" in text
    assert "雷丘" in text
