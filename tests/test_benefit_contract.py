"""Контракт benefit-aware каскада 1.x: что именно обещано и как это проверяется.

Контракт описан в `docs/stability.md` §8 и в README (раздел benefit-эскалации).
Тест держит обещание: если поверхность меняется без major, падает не только
документация, но и этот файл.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.backends.base import BackendResult, Capabilities
from src.backends.benefit import BenefitAwareCascadeBackend
from src.sampler.questions import NoulSpec
from src.schemas import NoulResult

ROOT = Path(__file__).resolve().parents[1]

#: Переменные окружения, замороженные контрактом 1.x.
ENV_VARS = (
    "DECISION_CASCADE_POLICY",
    "DECISION_CASCADE_REMOTE_ACCURACY",
    "DECISION_CASCADE_REMOTE_ACCURACY_BOUND",
    "DECISION_CASCADE_ESCALATION_COST",
    "DECISION_CASCADE_ERROR_COST",
    "DECISION_CASCADE_FEEDBACK_SHRINKAGE",
)

#: Ключи `/info -> engine`, которые обещаны при включённом benefit-каскаде.
INFO_KEYS = (
    "policy",
    "remote_accuracy",
    "remote_accuracy_default",
    "escalation_cost",
    "error_cost",
    "feedback_shrinkage",
    "benefit_threshold",
    "benefit_thresholds",
    "benefit_feedback",
)

#: Словарь `usage.reason` benefit-каскада.
USAGE_REASONS = {"benefit"}


class _StubBackend:
    def __init__(self, name: str, decision_map: dict[str, float]):
        self.name = name
        self.decision_map = decision_map
        self.capabilities = Capabilities(modalities=frozenset({"text"}))
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    def decide(self, state, questions, *, images=None, videos=None, media_kwargs=None):
        self.calls.append((state, tuple(questions)))
        decisions = {
            name: NoulResult(noul=self.decision_map[name], confidence=self.decision_map[name])
            for name in questions
        }
        return BackendResult(decisions=decisions, usage={"calls": 1})

    def close(self) -> None:
        pass


def _questions(names):
    return {name: NoulSpec(f"?{name}") for name in names}


def _cascade(remote_accuracy=0.9, **kwargs) -> BenefitAwareCascadeBackend:
    return BenefitAwareCascadeBackend(
        _StubBackend("local", {"a": 0.95, "b": 0.40}),
        _StubBackend("remote", {"a": 0.0, "b": 0.99}),
        remote_accuracy,
        **kwargs,
    )


def test_contract_is_documented():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    stability = (ROOT / "docs" / "stability.md").read_text(encoding="utf-8")
    for name in ENV_VARS:
        assert name in readme, f"{name} не описан в README"
        assert name in stability, f"{name} не описан в docs/stability.md"
    for key in INFO_KEYS:
        assert key in stability, f"{key} не описан в docs/stability.md"
    for reason in USAGE_REASONS:
        assert f'"{reason}"' in stability
    assert "Контракт benefit-каскада" in stability
    # статус поверхности: стабильна, а не beta
    row = next(
        line for line in stability.splitlines() if line.startswith("| Benefit-каскад")
    )
    assert "стабильно" in row


def test_info_surface(monkeypatch):
    from api.server.main import _engine_info, create_cascade_backend
    from api.server.service import DecisionService

    monkeypatch.setenv("DECISION_CASCADE_FALLBACK_URL", "http://fallback")
    monkeypatch.setenv("DECISION_CASCADE_POLICY", "benefit")
    monkeypatch.setenv("DECISION_CASCADE_REMOTE_ACCURACY", "0.86")
    monkeypatch.setenv("DECISION_CASCADE_ESCALATION_COST", "0.1")
    monkeypatch.setenv("DECISION_CASCADE_ERROR_COST", "1.0")
    monkeypatch.setenv("DECISION_CASCADE_FEEDBACK_SHRINKAGE", "50")
    monkeypatch.delenv("DECISION_CASCADE_RISK_BUDGET", raising=False)

    info = _engine_info(create_cascade_backend(DecisionService(seed=0)))
    assert info is not None
    missing = set(INFO_KEYS) - set(info)
    assert not missing
    assert info["policy"] == "benefit"
    assert info["remote_accuracy"] == pytest.approx(0.86)
    assert info["remote_accuracy_default"] == pytest.approx(0.86)
    assert info["escalation_cost"] == pytest.approx(0.1)
    assert info["error_cost"] == pytest.approx(1.0)
    assert info["feedback_shrinkage"] == pytest.approx(50.0)
    # q - C/(1+D) = 0.86 - 0.05
    assert info["benefit_threshold"] == pytest.approx(0.81)
    assert info["benefit_thresholds"] == {}
    assert info["benefit_feedback"] == {}


def test_info_surface_per_question(tmp_path, monkeypatch):
    from api.server.main import _engine_info, create_cascade_backend
    from api.server.service import DecisionService

    q_path = tmp_path / "q.json"
    q_path.write_text(
        '{"accuracy": 0.6, "per_question": {"topic": 0.8, "business": 0.4}}',
        encoding="utf-8",
    )
    monkeypatch.setenv("DECISION_CASCADE_FALLBACK_URL", "http://fallback")
    monkeypatch.setenv("DECISION_CASCADE_POLICY", "benefit")
    monkeypatch.setenv("DECISION_CASCADE_REMOTE_ACCURACY", str(q_path))
    monkeypatch.setenv("DECISION_CASCADE_ESCALATION_COST", "0.0")
    monkeypatch.delenv("DECISION_CASCADE_RISK_BUDGET", raising=False)

    info = _engine_info(create_cascade_backend(DecisionService(seed=0)))
    assert info is not None
    assert info["remote_accuracy"] == {
        "topic": pytest.approx(0.8),
        "business": pytest.approx(0.4),
    }
    # средняя q для вопроса вне словаря
    assert info["remote_accuracy_default"] == pytest.approx(0.6)
    assert info["benefit_thresholds"] == {
        "business": pytest.approx(0.4),
        "topic": pytest.approx(0.8),
    }


def test_usage_reason_is_a_documented_word():
    cascade = _cascade()
    result = cascade.decide("s", _questions(["a", "b"]))
    assert result.usage["reason"] in USAGE_REASONS
    assert result.usage["reason"] == "benefit"
    assert result.usage["escalated"] == 1


def test_benefit_takes_precedence_over_risk_budget(monkeypatch, tmp_path):
    """Если заданы обе политики, работает benefit — и risk.json не нужен."""
    from api.server.main import create_cascade_backend
    from api.server.service import DecisionService

    monkeypatch.setenv("DECISION_CASCADE_FALLBACK_URL", "http://fallback")
    monkeypatch.setenv("DECISION_CASCADE_POLICY", "benefit")
    monkeypatch.setenv("DECISION_CASCADE_REMOTE_ACCURACY", "0.86")
    monkeypatch.setenv("DECISION_CASCADE_RISK_BUDGET", "0.05")
    # файла нет: risk-каскад упал бы на старте, benefit — нет
    monkeypatch.setenv("DECISION_RISK_MODEL_PATH", str(tmp_path / "missing.json"))

    backend = create_cascade_backend(DecisionService(seed=0))
    assert isinstance(backend, BenefitAwareCascadeBackend)
    assert backend.policy == "benefit"


def test_unknown_policy_is_a_startup_error(monkeypatch):
    from api.server.main import create_cascade_backend
    from api.server.service import DecisionService

    monkeypatch.setenv("DECISION_CASCADE_FALLBACK_URL", "http://fallback")
    monkeypatch.setenv("DECISION_CASCADE_POLICY", "best-effort")
    monkeypatch.delenv("DECISION_CASCADE_RISK_BUDGET", raising=False)

    with pytest.raises(RuntimeError, match="DECISION_CASCADE_POLICY"):
        create_cascade_backend(DecisionService(seed=0))


def test_invalid_accuracy_bound_is_a_startup_error(monkeypatch):
    from api.server.main import create_cascade_backend
    from api.server.service import DecisionService

    monkeypatch.setenv("DECISION_CASCADE_FALLBACK_URL", "http://fallback")
    monkeypatch.setenv("DECISION_CASCADE_POLICY", "benefit")
    monkeypatch.setenv("DECISION_CASCADE_REMOTE_ACCURACY", "0.86")
    monkeypatch.setenv("DECISION_CASCADE_REMOTE_ACCURACY_BOUND", "optimistic")
    monkeypatch.delenv("DECISION_CASCADE_RISK_BUDGET", raising=False)

    with pytest.raises(RuntimeError, match="DECISION_CASCADE_REMOTE_ACCURACY_BOUND"):
        create_cascade_backend(DecisionService(seed=0))


def test_feedback_response_contract():
    from api.server import main

    cascade = _cascade()
    result = cascade.decide("s", _questions(["b"]))
    original = main.backend
    main.backend = cascade
    try:
        with TestClient(main.app) as client:
            response = client.post(
                "/v1/feedback",
                json={
                    "examples": [
                        {
                            "question": "b",
                            "decision": {
                                "type": "noul",
                                "noul": 0.99,
                                "confidence": 0.99,
                            },
                            "correct": True,
                        }
                    ]
                },
            )
            assert response.status_code == 200
            body = response.json()
            row = body["questions"][0]
            assert set(row) == {
                "question",
                "accepted",
                "examples",
                "reference",
                "retrained",
                "remote_accuracy",
            }
            assert row["examples"] == 1
            assert row["retrained"] is False
            assert row["remote_accuracy"] == pytest.approx(
                cascade.remote_accuracy_for("b")
            )
    finally:
        main.backend = original
    assert result.usage["escalated"] == 1


def test_feedback_unsupported_message_mentions_benefit(monkeypatch):
    from api.server import main
    from src.backends.cascade import CascadeBackend

    monkeypatch.setattr(
        main,
        "backend",
        CascadeBackend(
            _StubBackend("local", {"a": 0.5}),
            _StubBackend("remote", {"a": 0.5}),
            threshold=0.6,
        ),
    )
    with TestClient(main.app) as client:
        response = client.post(
            "/v1/feedback",
            json={
                "examples": [
                    {
                        "question": "a",
                        "decision": {"type": "noul", "noul": 0.5, "confidence": 0.5},
                        "correct": True,
                    }
                ]
            },
        )
    assert response.status_code == 409
    message = response.json()["error"]["message"]
    assert "benefit" in message
