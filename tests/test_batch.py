"""Тесты батч-принятия решений — написаны до реализации (RED)."""
import pytest
from fastapi.testclient import TestClient

from api.server.service import DecisionService
from src.sampler.questions import ChoiceSpec, NoulSpec, ScoreSpec

QUESTIONS = {
    "срочность": NoulSpec("Срочно?"),
    "команда": ChoiceSpec("Кто?", ("биллинг", "интеграция")),
    "накал": ScoreSpec("Накал", ("низкий", "средний", "высокий")),
}
STATES = ["Stripe не работает, теряем заказы", "Как поменять тариф?"]


def test_decide_many_matches_sequential():
    svc = DecisionService(seed=0)
    batch = svc.decide_many(STATES, QUESTIONS)
    assert len(batch) == len(STATES)
    for state, batched in zip(STATES, batch):
        single = svc.decide(state, QUESTIONS)
        assert set(batched) == set(single)
        for name in single:
            assert batched[name] == single[name]


def test_decide_many_empty_states():
    svc = DecisionService(seed=0)
    assert svc.decide_many([], QUESTIONS) == []


def test_decide_many_empty_questions():
    svc = DecisionService(seed=0)
    assert svc.decide_many(STATES, {}) == [{}, {}]


@pytest.fixture(scope="module")
def client():
    from api.server.main import app

    with TestClient(app) as c:
        yield c


BATCH_PAYLOAD = {
    "states": STATES,
    "questions": {
        "срочность": {"type": "noul", "instructions": "Срочно?"},
        "команда": {
            "type": "choice",
            "instructions": "Кто?",
            "options": ["биллинг", "интеграция"],
        },
    },
}


def test_batch_endpoint_matches_single(client):
    resp = client.post("/decisions/batch", json=BATCH_PAYLOAD)
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["results"]) == len(STATES)
    for state, result in zip(STATES, body["results"]):
        single = client.post(
            "/decisions", json={"state": state, "questions": BATCH_PAYLOAD["questions"]}
        ).json()["decisions"]
        assert set(result) == set(single)


def test_batch_endpoint_rejects_empty_states(client):
    resp = client.post("/decisions/batch", json={**BATCH_PAYLOAD, "states": []})
    assert resp.status_code == 422
