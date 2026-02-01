from __future__ import annotations


def test_pokedex_shortcut_evolution_is_concise_chat_style():
    """
    When the user asks a simple evolution question, we should answer in a
    conversation-friendly way (no full pokedex card dump).
    """
    from src.agents.pokedex_shortcut import maybe_answer_pokedex

    ans = maybe_answer_pokedex("胖丁进化什么")
    assert ans is not None

    text = ans.content
    assert "胖丁" in text
    assert "胖可丁" in text
    assert "宝宝丁" in text

    # Keep it concise by default: do not include unrelated sections unless asked.
    assert "属性" not in text
    assert "身高" not in text
    assert "体重" not in text
    assert "特性" not in text
    assert "##" not in text


def test_pokedex_shortcut_detail_request_includes_basic_info():
    """
    If the user explicitly asks for the pokedex/detailed info, include basics.
    """
    from src.agents.pokedex_shortcut import maybe_answer_pokedex

    ans = maybe_answer_pokedex("胖丁图鉴")
    assert ans is not None

    text = ans.content
    assert "胖丁" in text
    assert "属性" in text
    assert "身高" in text
    assert "体重" in text
