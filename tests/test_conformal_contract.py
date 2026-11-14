"""Контракт conformal-каскада 1.x: сертификат, `/info`, `usage`, гейт.

Контракт описан в `docs/stability.md` §9 и в README. Тест держит обещание:
если поверхность меняется без major, падают и документация, и этот файл.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.backends.base import BackendResult, Capabilities
from src.backends.conformal import ConformalCascadeBackend
from src.conformal import METHOD
from src.risk import LEGACY_FEATURE_NAMES, ErrorRiskModel
from src.sampler.questions import NoulSpec
from src.schemas import NoulResult

ROOT = Path(__file__).resolve().parents[1]

#: Переменные окружения, замороженные контрактом 1.x.
ENV_VARS = (
    "DECISION_CASCADE_POLICY",
    "DECISION_CONFORMAL_PATH",
    "DECISION_RISK_MODEL_PATH",
)

#: Ключи `/info -> engine`, которые обещаны при conformal-каскаде.
INFO_KEYS = (
    "policy",
    "conformal_alpha",
    "conformal_delta",
    "conformal_threshold",
    "conformal_coverage",
    "conformal_risk",
    "conformal_upper_bound",
    "conformal_calibration_n",
    "conformal_gate",
    "conformal_gate_worse",
)

#: Словарь `usage.reason` conformal-каскада.
USAGE_REASONS = {"conformal", "conformal_gate"}


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


def _risk_model() -> ErrorRiskModel:
    # z = -20*confidence + 14: уверенно -> P(ошибка) ~0, у границы 0.55 -> ~0.95
    return ErrorRiskModel(
        weights=np.array([-20.0, 0.0, 0.0]),
        bias=14.0,
        feature_names=LEGACY_FEATURE_NAMES,
    )


def _certificate(**overrides):
    certificate = {
        "schema_version": "1",
        "method": METHOD,
        "score": "p_error",
        "alpha": 0.05,
        "delta": 0.05,
        "n": 200,
        "min_coverage": 0.5,
        "threshold": 0.5,
        "accepted": 150,
        "coverage": 0.75,
        "empirical_risk": 0.02,
        "upper_bound": 0.04,
        "gate": {
            "accepted": True,
            "baseline": "confidence",
            "tol": 0.01,
            "worse": [],
            "test": {
                "n": 100,
                "coverage": 0.7,
                "risk": 0.03,
                "accuracy": 0.97,
                "baseline_risk": 0.05,
            },
        },
    }
    certificate.update(overrides)
    return certificate


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
    assert "Контракт conformal-каскада" in stability
    row = next(
        line for line in stability.splitlines() if line.startswith("| Conformal-каскад")
    )
    assert "стабильно" in row


def test_info_surface():
    from api.server.main import _engine_info

    backend = ConformalCascadeBackend(
        _StubBackend("local", {"a": 0.9}),
        _StubBackend("remote", {"a": 0.9}),
        _risk_model(),
        _certificate(),
    )
    info = _engine_info(backend)
    assert info is not None
    missing = set(INFO_KEYS) - set(info)
    assert not missing
    assert info["policy"] == "conformal"
    assert info["conformal_alpha"] == pytest.approx(0.05)
    assert info["conformal_threshold"] == pytest.approx(0.5)
    assert info["conformal_calibration_n"] == 200
    assert info["conformal_gate"] == "ok"
    assert info["conformal_test"]["risk"] == pytest.approx(0.03)
    # риск-ключи не должны протекать в /info conformal-каскада
    assert "risk_budget" not in info


def test_usage_reason_is_a_documented_word():
    cascade = ConformalCascadeBackend(
        _StubBackend("local", {"a": 0.9, "b": 0.55}),
        _StubBackend("remote", {"a": 0.0, "b": 0.99}),
        _risk_model(),
        _certificate(),
    )
    result = cascade.decide("s", _questions(["a", "b"]))
    assert result.usage["reason"] in USAGE_REASONS
    assert result.usage["reason"] == "conformal"
    assert result.usage["escalated"] == 1
    assert cascade.fallback.calls == [("s", ("b",))]


def test_rejected_certificate_falls_back_to_confidence():
    certificate = _certificate(
        gate={"accepted": False, "worse": ["selective risk выше alpha"], "test": {}}
    )
    cascade = ConformalCascadeBackend(
        _StubBackend("local", {"a": 0.9, "b": 0.55}),
        _StubBackend("remote", {"a": 0.0, "b": 0.99}),
        _risk_model(),
        certificate,
        fallback_threshold=0.6,
    )
    result = cascade.decide("s", _questions(["a", "b"]))
    assert result.usage["reason"] == "conformal_gate"
    assert result.usage["escalated"] == 1
    assert cascade.fallback.calls == [("s", ("b",))]


def test_conformal_does_not_accept_feedback():
    cascade = ConformalCascadeBackend(
        _StubBackend("local", {"a": 0.9}),
        _StubBackend("remote", {"a": 0.9}),
        _risk_model(),
        _certificate(),
    )
    assert cascade.capabilities.accepts_feedback is False


def test_conformal_takes_precedence_over_risk_budget(monkeypatch, tmp_path):
    from api.server.main import create_cascade_backend
    from api.server.service import DecisionService

    model_path = tmp_path / "risk.json"
    _risk_model().save(model_path)
    cert_path = tmp_path / "conformal.json"
    cert_path.write_text(json.dumps(_certificate()), encoding="utf-8")
    monkeypatch.setenv("DECISION_CASCADE_FALLBACK_URL", "http://fallback")
    monkeypatch.setenv("DECISION_CASCADE_POLICY", "conformal")
    monkeypatch.setenv("DECISION_CONFORMAL_PATH", str(cert_path))
    monkeypatch.setenv("DECISION_RISK_MODEL_PATH", str(model_path))
    monkeypatch.setenv("DECISION_CASCADE_RISK_BUDGET", "0.05")

    backend = create_cascade_backend(DecisionService(seed=0))
    assert isinstance(backend, ConformalCascadeBackend)
    assert backend.policy == "conformal"


def test_missing_certificate_is_a_startup_error(monkeypatch, tmp_path):
    from api.server.main import create_cascade_backend
    from api.server.service import DecisionService

    model_path = tmp_path / "risk.json"
    _risk_model().save(model_path)
    monkeypatch.setenv("DECISION_CASCADE_FALLBACK_URL", "http://fallback")
    monkeypatch.setenv("DECISION_CASCADE_POLICY", "conformal")
    monkeypatch.setenv("DECISION_CONFORMAL_PATH", str(tmp_path / "missing.json"))
    monkeypatch.setenv("DECISION_RISK_MODEL_PATH", str(model_path))
    monkeypatch.delenv("DECISION_CASCADE_RISK_BUDGET", raising=False)

    with pytest.raises(RuntimeError, match="conformal"):
        create_cascade_backend(DecisionService(seed=0))


def test_missing_risk_model_is_a_startup_error(monkeypatch, tmp_path):
    from api.server.main import create_cascade_backend
    from api.server.service import DecisionService

    cert_path = tmp_path / "conformal.json"
    cert_path.write_text(json.dumps(_certificate()), encoding="utf-8")
    monkeypatch.setenv("DECISION_CASCADE_FALLBACK_URL", "http://fallback")
    monkeypatch.setenv("DECISION_CASCADE_POLICY", "conformal")
    monkeypatch.setenv("DECISION_CONFORMAL_PATH", str(cert_path))
    monkeypatch.setenv("DECISION_RISK_MODEL_PATH", str(tmp_path / "missing.json"))
    monkeypatch.delenv("DECISION_CASCADE_RISK_BUDGET", raising=False)

    with pytest.raises(RuntimeError, match="risk model"):
        create_cascade_backend(DecisionService(seed=0))


def test_feedback_unsupported_message_mentions_conformal():
    from api.server import main

    cascade = ConformalCascadeBackend(
        _StubBackend("local", {"a": 0.9}),
        _StubBackend("remote", {"a": 0.9}),
        _risk_model(),
        _certificate(),
    )
    original = main.backend
    main.backend = cascade
    try:
        with TestClient(main.app) as client:
            response = client.post(
                "/v1/feedback",
                json={
                    "examples": [
                        {
                            "question": "a",
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
    finally:
        main.backend = original
    assert response.status_code == 409
    assert "conformal" in response.json()["error"]["message"]
