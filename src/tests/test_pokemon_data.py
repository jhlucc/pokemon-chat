from __future__ import annotations


def test_pokemon_data_load_and_lookup():
    from src.agents.pokemon_data import get_pokemon_data

    data = get_pokemon_data()

    pikachu = data.get_by_cn_name("皮卡丘")
    assert pikachu is not None
    assert pikachu["id"] == 25
    assert "电" in (pikachu.get("type") or [])

    by_id = data.get_by_id(25)
    assert by_id is not None
    assert by_id.get("chinese_name") == "皮卡丘"


def test_pokemon_data_has_expected_size():
    from src.agents.pokemon_data import get_pokemon_data

    data = get_pokemon_data()
    assert len(list(data.iter_all())) >= 800


def test_pokemon_name_alias_resolution():
    from src.agents.pokemon_data import get_pokemon_data

    data = get_pokemon_data()

    assert data.resolve_name("Pikachu") == "皮卡丘"
    assert data.resolve_name("ピカチュウ") == "皮卡丘"
