"""Метрики Prometheus: счётчики запросов, решений, готовности и /metrics."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from prometheus_client import REGISTRY
from prometheus_client.parser import text_string_to_metric_families

from api.server import metrics

PAYLOAD = {
    "state": "Stripe failing for 3 days",
    "questions": {"urgent": {"type": "noul", "instructions": "urgent?"}},
}


@pytest.fixture(scope="module")
def client():
    from api.server.main import app

    with TestClient(app) as c:
        yield c


def _value(name: str, **labels: str) -> float:
    """Значение метрики из реестра; 0.0, если её ещё не было."""
    return REGISTRY.get_sample_value(name, labels) or 0.0


def test_metrics_endpoint_returns_prometheus_text(client):
    response = client.get("/metrics")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert "decision_http_requests_total" in response.text
    assert "decision_ready" in response.text


def test_metrics_endpoint_is_not_self_counted(client):
    client.get("/metrics")
    client.get("/metrics")
    assert _value(
        "decision_http_requests_total",
        method="GET",
        path="/metrics",
        status="200",
    ) == 0.0


def test_requests_are_counted_by_route_template(client):
    before = _value(
        "decision_http_requests_total",
        method="GET",
        path="/health",
        status="200",
    )
    client.get("/health")
    client.get("/health")
    after = _value(
        "decision_http_requests_total",
        method="GET",
        path="/health",
        status="200",
    )
    assert after - before == 2.0


def test_decisions_are_counted_with_question_and_type(client):
    before = _value(
        "decision_decisions_total",
        question="urgent",
        type="noul",
        trained="false",
    )
    response = client.post("/decisions", json=PAYLOAD)
    assert response.status_code == 200
    trained = "true" if response.json()["trained"] else "false"
    after = _value(
        "decision_decisions_total",
        question="urgent",
        type="noul",
        trained=trained,
    )
    assert after > before


def test_batch_counts_every_state(client):
    before = _value(
        "decision_decisions_total",
        question="urgent",
        type="noul",
        trained="false",
    )
    payload = {"states": ["a", "b", "c"], "questions": PAYLOAD["questions"]}
    response = client.post("/decisions/batch", json=payload)
    assert response.status_code == 200
    trained = "true" if response.json()["trained"] else "false"
    after = _value(
        "decision_decisions_total",
        question="urgent",
        type="noul",
        trained=trained,
    )
    assert after - before == 3.0


def test_readiness_gauge_tracks_response(client):
    response = client.get("/ready")
    expected = 1.0 if response.status_code == 200 else 0.0
    assert _value("decision_ready") == expected


def test_untrained_refusal_is_counted():
    from src.errors import UntrainedDecisionError

    before = _value("decision_untrained_refusals_total")
    from api.server.main import _untrained_response

    response = _untrained_response(UntrainedDecisionError("abc", "checkpoints"))
    assert response.status_code == 409
    assert _value("decision_untrained_refusals_total") - before == 1.0


def test_route_template_falls_back_to_path_for_unknown_route():
    class _Request:
        url = type("URL", (), {"path": "/unknown"})()

        def __init__(self) -> None:
            self.scope: dict = {}

    assert metrics.route_template(_Request()) == "/unknown"


def test_rendered_metrics_parse():
    text = metrics.render().decode()
    families = list(text_string_to_metric_families(text))
    names = {family.name for family in families}
    # prometheus_client отдаёт counter как <name>_total, парсер — без суффикса
    assert "decision_http_requests" in names
    assert "decision_decisions" in names
    assert "decision_ready" in names
