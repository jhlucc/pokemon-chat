from __future__ import annotations


def test_should_web_search_time_sensitive_query():
    from src.agents.web_gating import should_web_search

    assert should_web_search("宝可梦最新活动") is True


def test_should_not_web_search_static_pokedex_query():
    from src.agents.web_gating import should_web_search

    assert should_web_search("皮卡丘属性") is False
