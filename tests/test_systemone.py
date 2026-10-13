"""Тесты стандартного протокола Jev/SystemOne: /v1/systemone."""
import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    from api.server.main import app

    with TestClient(app) as c:
        yield c


PAYLOAD = {
    "model": "local",
    "state": "Stripe failing for 3 days, I am losing sales",
    "questions": {
        "urgent": {"type": "noul", "instructions": "Is this urgent?"},
        "team": {
            "type": "choice",
            "instructions": "Who should handle it?",
            "criteria": {"billing": "money", "technical": "errors"},
        },
        "severity": {
            "type": "score",
            "instructions": "How severe?",
            "criteria": ["none", "minor", "critical"],
        },
    },
}


def test_systemone_envelope(client):
    resp = client.post("/v1/systemone", json=PAYLOAD)
    assert resp.status_code == 200
    body = resp.json()
    assert {"model", "answers", "usage"} <= set(body)
    assert set(body["answers"]) == {"urgent", "team", "severity"}


def test_systemone_answer_types(client):
    answers = client.post("/v1/systemone", json=PAYLOAD).json()["answers"]
    assert answers["urgent"]["type"] == "noul"
    assert 0.0 <= answers["urgent"]["probability"] <= 1.0
    assert "confidence" in answers["urgent"]

    assert answers["team"]["type"] == "choice"
    assert answers["team"]["choice"] in {"billing", "technical"}
    assert abs(sum(answers["team"]["probabilities"].values()) - 1.0) < 1e-6

    assert answers["severity"]["type"] == "score"
    assert answers["severity"]["legend"] == ["none", "minor", "critical"]
    assert abs(sum(answers["severity"]["probabilities"].values()) - 1.0) < 1e-6


def test_systemone_choice_options_derived_from_criteria(client):
    payload = {
        "state": "s",
        "questions": {"q": {"type": "choice", "criteria": {"a": "A", "b": "B"}}},
    }
    resp = client.post("/v1/systemone", json=payload)
    assert resp.status_code == 200
    assert set(resp.json()["answers"]["q"]["probabilities"]) == {"a", "b"}


def test_systemone_score_levels_derived_from_criteria(client):
    payload = {
        "state": "s",
        "questions": {"q": {"type": "score", "criteria": ["low", "high"]}},
    }
    resp = client.post("/v1/systemone", json=payload)
    assert resp.status_code == 200
    assert resp.json()["answers"]["q"]["legend"] == ["low", "high"]


def test_systemone_instructions_optional(client):
    payload = {"state": "s", "questions": {"myq": {"type": "noul"}}}
    resp = client.post("/v1/systemone", json=payload)
    assert resp.status_code == 200
    assert resp.json()["answers"]["myq"]["type"] == "noul"


def test_systemone_rejects_unsupported_image(client):
    payload = {
        "state": "s",
        "images": ["base64data"],
        "questions": {"q": {"type": "noul", "instructions": "?"}},
    }
    resp = client.post("/v1/systemone", json=payload)
    assert resp.status_code == 422
    error = resp.json()["error"]
    assert error["code"] == "MODALITY_NOT_SUPPORTED"
    assert "image" in error["requested"]
    assert "image" not in error["supported"]


def test_systemone_rejects_duplicate_choice_options(client):
    payload = {
        "state": "s",
        "questions": {"q": {"type": "choice", "options": ["a", "a"]}},
    }
    resp = client.post("/v1/systemone", json=payload)
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "INVALID_QUESTION"


def test_systemone_rejects_duplicate_score_levels(client):
    payload = {
        "state": "s",
        "questions": {"q": {"type": "score", "criteria": ["a", "a"]}},
    }
    resp = client.post("/v1/systemone", json=payload)
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "INVALID_QUESTION"


def test_systemone_rejects_score_criteria_length_mismatch(client):
    payload = {
        "state": "s",
        "questions": {
            "q": {"type": "score", "levels": ["a", "b", "c"], "criteria": ["x", "y"]}
        },
    }
    resp = client.post("/v1/systemone", json=payload)
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "INVALID_QUESTION"


def test_systemone_accepts_images_with_vision_backend(monkeypatch):
    from api.server import main
    from api.server.service import DecisionService
    from src.backends.local import LocalHeadsBackend
    from src.encoder.vision import HashVisionEncoder

    service = DecisionService(seed=0, vision_encoder=HashVisionEncoder(dim=64))
    monkeypatch.setattr(main, "backend", LocalHeadsBackend(service))
    payload = {
        "state": "s",
        "images": ["a screenshot"],
        "questions": {"q": {"type": "noul", "instructions": "?"}},
    }
    with TestClient(main.app) as tc:
        resp = tc.post("/v1/systemone", json=payload)
    assert resp.status_code == 200


def test_systemone_rejects_choice_without_options_or_criteria(client):
    payload = {"state": "s", "questions": {"q": {"type": "choice"}}}
    resp = client.post("/v1/systemone", json=payload)
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "INVALID_QUESTION"
