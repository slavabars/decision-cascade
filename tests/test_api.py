"""Интеграционные тесты API — написаны до реализации сервиса (TDD RED)."""
import pytest
from fastapi.testclient import TestClient

from src.schemas import SCHEMA_VERSION


@pytest.fixture(scope="module")
def client():
    from api.server.main import app

    with TestClient(app) as c:
        yield c


PAYLOAD = {
    "state": "Stripe connection failing for 3 days, I am losing sales",
    "questions": {
        "is_urgent": {"type": "noul", "instructions": "urgent?"},
        "department": {
            "type": "choice",
            "instructions": "who?",
            "options": ["billing", "technical", "sales"],
        },
        "frustration": {
            "type": "score",
            "instructions": "how bad?",
            "levels": ["calm", "mild", "moderate", "high", "extreme"],
        },
    },
}


def test_decisions_returns_envelope(client):
    resp = client.post("/decisions", json=PAYLOAD)
    assert resp.status_code == 200
    body = resp.json()
    assert body["schema_version"] == SCHEMA_VERSION
    assert set(body["decisions"]) == {"is_urgent", "department", "frustration"}


def test_decisions_typed_contract(client):
    body = client.post("/decisions", json=PAYLOAD).json()["decisions"]
    n, c, s = body["is_urgent"], body["department"], body["frustration"]
    assert n["type"] == "noul" and 0.0 <= n["noul"] <= 1.0
    assert "confidence" in n and 0.0 <= n["confidence"] <= 1.0
    assert c["type"] == "choice" and c["choice"] in {"billing", "technical", "sales"}
    assert "confidence" in c and "probabilities" in c
    assert abs(sum(c["probabilities"].values()) - 1.0) < 1e-6
    assert s["type"] == "score" and s["legend"] == ["calm", "mild", "moderate", "high", "extreme"]
    assert "confidence" in s
    assert abs(sum(s["probabilities"].values()) - 1.0) < 1e-6


def test_decisions_no_raw_logits(client):
    body = client.post("/decisions", json=PAYLOAD).json()["decisions"]
    for res in body.values():
        assert "logits" not in res and "logit" not in res


def test_different_states_differ(client):
    a = client.post("/decisions", json=PAYLOAD).json()["decisions"]
    other = dict(PAYLOAD)
    other["state"] = "I want to talk about the weather and a gift for my dog"
    b = client.post("/decisions", json=other).json()["decisions"]
    assert a["is_urgent"]["noul"] != b["is_urgent"]["noul"]


def test_unknown_question_type_rejected(client):
    payload = {
        "state": "s",
        "questions": {"q": {"type": "frobnicate", "instructions": "?"}},
    }
    resp = client.post("/decisions", json=payload)
    assert resp.status_code == 422


def test_single_option_choice_rejected(client):
    payload = {
        "state": "s",
        "questions": {"q": {"type": "choice", "instructions": "?", "options": ["only"]}},
    }
    resp = client.post("/decisions", json=payload)
    assert resp.status_code == 422


def test_empty_questions_rejected(client):
    resp = client.post("/decisions", json={"state": "s", "questions": {}})
    assert resp.status_code == 422


def test_choice_criteria_accepted(client):
    payload = {
        "state": "s",
        "questions": {
            "dept": {
                "type": "choice",
                "instructions": "who?",
                "options": ["billing", "sales"],
                "criteria": {"billing": "money", "sales": "deals"},
            }
        },
    }
    resp = client.post("/decisions", json=payload)
    assert resp.status_code == 200
    assert resp.json()["decisions"]["dept"]["type"] == "choice"


def test_to_specs_carries_criteria():
    from api.server.main import ChoiceQuestion, ScoreQuestion, _to_specs

    specs = _to_specs(
        {
            "dept": ChoiceQuestion(
                type="choice",
                instructions="who?",
                options=["a", "b"],
                criteria={"a": "Alpha", "b": "Beta"},
            ),
            "pain": ScoreQuestion(
                type="score",
                instructions="how bad?",
                levels=["low", "high"],
                criteria=["calm", "critical"],
            ),
        }
    )
    assert specs["dept"].criteria == {"a": "Alpha", "b": "Beta"}
    assert specs["pain"].criteria == ("calm", "critical")