from __future__ import annotations


def test_classify_intent_pokedex_facts():
    from src.agents.intent import Intent, classify_intent

    decision = classify_intent("皮卡丘 属性")
    assert decision.intent == Intent.POKEDEX_FACTS


def test_classify_intent_team_building():
    from src.agents.intent import Intent, classify_intent

    decision = classify_intent("给我一套队伍")
    assert decision.intent == Intent.TEAM_BUILDING


def test_classify_intent_web_search():
    from src.agents.intent import Intent, classify_intent

    decision = classify_intent("宝可梦 最新 活动")
    assert decision.intent == Intent.WEB_SEARCH


def test_classify_intent_asks_clarification_when_missing_entity():
    from src.agents.intent import Intent, classify_intent

    decision = classify_intent("属性相性怎么查？")
    assert decision.intent == Intent.POKEDEX_FACTS
    assert decision.needs_clarification is True
    assert decision.clarification_question
    assert "宝可梦" in decision.clarification_question
