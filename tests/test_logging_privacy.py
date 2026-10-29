"""Приватность логов: state, images и тела /v1/feedback не попадают в лог."""
import logging

import pytest
from fastapi.testclient import TestClient

from src.backends.base import BackendResult, Capabilities
from src.sampler.questions import NoulSpec


@pytest.fixture(scope="module")
def client():
    from api.server.main import app

    with TestClient(app) as c:
        yield c


def test_service_logs_decision_but_not_state(caplog):
    from api.server.service import DecisionService

    marker = "UNIQUE_STATE_MARKER_9f3a"
    service = DecisionService(seed=0)
    with caplog.at_level(logging.DEBUG, logger="decision.service"):
        service.decide(marker, {"urgent": NoulSpec("urgent?")})

    assert marker not in caplog.text
    assert any("decision name=urgent" in record.getMessage() for record in caplog.records)


def test_decisions_endpoint_does_not_log_state(client, caplog):
    marker = "UNIQUE_API_STATE_MARKER_7c1e"
    with caplog.at_level(logging.DEBUG):
        resp = client.post(
            "/decisions",
            json={
                "state": marker,
                "questions": {"urgent": {"type": "noul", "instructions": "?"}},
            },
        )
    assert resp.status_code == 200
    assert marker not in caplog.text


class _FeedbackBackend:
    name = "feedback"
    capabilities = Capabilities(accepts_feedback=True)

    def __init__(self):
        self.example_counts: dict[str, int] = {}

    def decide(self, state, questions, **kwargs):
        return BackendResult(decisions={}, usage={})

    def add_examples(self, question, decisions, correct):
        self.example_counts[question] = self.example_counts.get(question, 0) + len(
            decisions
        )


def test_feedback_endpoint_does_not_log_examples(client, monkeypatch, caplog):
    from api.server import main

    marker = "UNIQUE_FEEDBACK_MARKER_31b0"
    monkeypatch.setattr(main, "backend", _FeedbackBackend())
    with caplog.at_level(logging.DEBUG):
        resp = client.post(
            "/v1/feedback",
            json={
                "examples": [
                    {
                        "question": marker,
                        "decision": {
                            "type": "noul",
                            "noul": 0.9,
                            "confidence": 0.9,
                        },
                        "correct": True,
                    }
                ]
            },
        )

    assert resp.status_code == 200
    assert resp.json()["questions"][0]["question"] == marker
    assert marker not in caplog.text


class _VisionBackend:
    name = "vision"
    capabilities = Capabilities(modalities=frozenset({"text", "image"}))

    def decide(self, state, questions, *, images=None, videos=None, media_kwargs=None):
        return BackendResult(decisions={}, usage={})


def test_systemone_does_not_log_image_paths(client, monkeypatch, caplog):
    from api.server import main

    marker = "UNIQUE_IMAGE_PATH_MARKER_5d2f"
    monkeypatch.setattr(main, "backend", _VisionBackend())
    with caplog.at_level(logging.DEBUG):
        resp = client.post(
            "/v1/systemone",
            json={
                "state": "text",
                "images": [f"/secret/{marker}.png"],
                "questions": {"urgent": {"type": "noul"}},
            },
        )

    assert resp.status_code == 200
    assert marker not in caplog.text
