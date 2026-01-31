from __future__ import annotations


def test_counter_team_fire_recommends_water():
    from src.agents.trainer_agent import counter_team

    text = counter_team.invoke({"opponent_types": ["火"]})
    assert "水" in text


def test_type_coverage_reports_unknown_types():
    from src.agents.trainer_agent import type_coverage

    text = type_coverage.invoke({"team_types": ["水", "电", "???"]})
    assert "未知属性" in text

