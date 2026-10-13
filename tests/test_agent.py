"""Тесты маршрутизации обращения (интеграция в чат-агента)."""
from api.sdk.agent import AgentAction, RoutingPolicy, route_ticket


def _noul(value: float) -> dict:
    return {"type": "noul", "noul": value, "confidence": max(value, 1 - value)}


def _choice(value: str, options) -> dict:
    return {
        "type": "choice",
        "choice": value,
        "probabilities": {o: (1.0 if o == value else 0.0) for o in options},
        "confidence": 1.0,
    }


def _score(value: float, levels) -> dict:
    return {
        "type": "score",
        "score": value,
        "probabilities": {level: 0.0 for level in levels},
        "confidence": 0.5,
        "legend": list(levels),
    }


TEAMS = ["биллинг", "интеграция", "общая поддержка", "продажи"]
HEAT = ["спокоен", "слегка недоволен", "умеренно", "сильно", "на грани"]
CHURN = ["точно останется", "вероятно останется", "неопределённо", "вероятно уйдёт", "уйдёт наверняка"]


def test_route_picks_team_from_choice():
    decisions = {"команда": _choice("биллинг", TEAMS)}
    assert route_ticket(decisions).team == "биллинг"


def test_route_defaults_team_when_missing():
    assert route_ticket({}).team == RoutingPolicy().default_team


def test_route_escalates_on_high_escalation_probability():
    decisions = {"эскалация": _noul(0.9)}
    action = route_ticket(decisions)
    assert action.escalate is True
    assert any("эскалация" in reason for reason in action.rationale)


def test_route_no_escalation_when_calm_and_low():
    decisions = {
        "срочность": _noul(0.1),
        "эскалация": _noul(0.1),
        "накал": _score(0.0, HEAT),
    }
    action = route_ticket(decisions)
    assert action.escalate is False
    assert action.priority == "low"


def test_route_urgent_priority_forces_escalation():
    decisions = {"срочность": _noul(0.95)}
    action = route_ticket(decisions)
    assert action.priority == "urgent"
    assert action.escalate is True


def test_route_high_priority_from_churn():
    decisions = {"срочность": _noul(0.1), "удержание": _score(3.5, CHURN)}
    action = route_ticket(decisions)
    assert action.priority == "high"
    assert action.tone == "эмпатичный"


def test_route_tone_reflects_anger():
    decisions = {"накал": _score(4.0, HEAT)}
    action = route_ticket(decisions)
    assert action.tone == "извиняющийся"
    assert "извинения" in action.reply.lower()


def test_route_reply_mentions_team_and_priority():
    decisions = {"команда": _choice("интеграция", TEAMS), "срочность": _noul(0.7)}
    action = route_ticket(decisions)
    assert "интеграция" in action.reply
    assert "urgent" in action.reply


def test_route_is_typed_action():
    action = route_ticket({"команда": _choice("продажи", TEAMS)})
    assert isinstance(action, AgentAction)
    assert isinstance(action.rationale, tuple)


def test_route_ignores_malformed_decisions():
    decisions = {"команда": {"type": "choice"}, "срочность": "oops"}
    action = route_ticket(decisions)
    assert action.team == RoutingPolicy().default_team
    assert action.priority in ("low", "normal", "high", "urgent")
