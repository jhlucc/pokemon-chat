from __future__ import annotations


def test_search_pokedex_uses_full_local_dataset():
    from src.agents.pokedex_agent import search_pokedex

    text = search_pokedex.invoke({"query": "电"})
    assert "皮卡丘" in text
    # Hardcoded mini-dex does not include 皮丘; full dataset does.
    assert "皮丘" in text


def test_get_evolution_chain_works_for_non_hardcoded_pokemon():
    from src.agents.pokedex_agent import get_evolution_chain

    text = get_evolution_chain.invoke({"pokemon_name": "波波"})
    assert "波波" in text
    assert "比比鸟" in text

