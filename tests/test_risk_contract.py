"""Контракт risk-aware каскада 1.x: что именно обещано и как это проверяется.

Контракт описан в `docs/stability.md` §7 и в README (таблица `DECISION_*`).
Тест держит обещание: если поверхность меняется без major, падает не только
документация, но и этот файл.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.backends.base import BackendResult, Capabilities
from src.backends.cascade import RiskAwareCascadeBackend
from src.risk import LEGACY_FEATURE_NAMES, ErrorRiskModel
from src.sampler.questions import NoulSpec
from src.schemas import NoulResult

ROOT = Path(__file__).resolve().parents[1]

#: Переменные окружения, замороженные контрактом 1.x.
ENV_VARS = (
    "DECISION_CASCADE_RISK_BUDGET",
    "DECISION_RISK_MODEL_PATH",
    "DECISION_RISK_FEW_SHOT_N",
    "DECISION_RISK_GATE_DELTA",
)

#: Ключи `/info -> engine`, которые обещаны при включённом risk-каскаде.
INFO_KEYS = (
    "risk_budget",
    "risk_gate",
    "risk_gate_delta",
    "risk_few_shot_n",
    "risk_few_shot",
)

#: Словарь `usage.reason` risk-каскада.
USAGE_REASONS = {"risk", "risk_gate"}


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
    # z = -20*confidence + 14: уверенно -> P(ошибка) ~0, у границы 0.5 -> ~0.95
    return ErrorRiskModel(
        weights=np.array([-20.0, 0.0, 0.0]),
        bias=14.0,
        feature_names=LEGACY_FEATURE_NAMES,
    )


def test_contract_is_documented():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    stability = (ROOT / "docs" / "stability.md").read_text(encoding="utf-8")
    for name in ENV_VARS:
        assert name in readme, f"{name} не описан в README"
        assert name in stability, f"{name} не описан в docs/stability.md"
    for key in INFO_KEYS:
        assert key in stability
    for reason in USAGE_REASONS:
        assert reason in stability
    assert "Контракт risk-aware каскада" in stability
    # статус поверхности: стабильна, а не beta
    assert "| Risk-aware каскад" in stability
    row = next(
        line for line in stability.splitlines() if line.startswith("| Risk-aware каскад")
    )
    assert "стабильно" in row


def test_info_surface(monkeypatch, tmp_path):
    from api.server.main import _engine_info, create_cascade_backend
    from api.server.service import DecisionService

    model_path = tmp_path / "risk.json"
    _risk_model().save(model_path)
    monkeypatch.setenv("DECISION_CASCADE_FALLBACK_URL", "http://fallback")
    monkeypatch.setenv("DECISION_CASCADE_RISK_BUDGET", "0.05")
    monkeypatch.setenv("DECISION_RISK_MODEL_PATH", str(model_path))
    monkeypatch.setenv("DECISION_RISK_FEW_SHOT_N", "10")
    monkeypatch.setenv("DECISION_RISK_GATE_DELTA", "0.0")

    info = _engine_info(create_cascade_backend(DecisionService(seed=0)))
    assert info is not None
    missing = set(INFO_KEYS) - set(info)
    assert not missing
    assert isinstance(info["risk_budget"], float)
    assert info["risk_gate"] in {"ok", "rejected"}
    assert isinstance(info["risk_gate_delta"], float)
    assert info["risk_few_shot_n"] == 10
    assert info["risk_few_shot"] == []


def test_usage_reason_is_a_documented_word():
    cascade = RiskAwareCascadeBackend(
        _StubBackend("local", {"a": 0.9, "b": 0.55}),
        _StubBackend("remote", {"a": 0.0, "b": 0.99}),
        _risk_model(),
        risk_budget=0.5,
    )
    result = cascade.decide("s", _questions(["a", "b"]))
    assert result.usage["reason"] in USAGE_REASONS
    assert result.usage["reason"] == "risk"
    assert result.usage["escalated"] == 1


def test_hard_gate_never_weakens_risk_rule():
    """Отклонённый сертификат: риск-правило не применяется без явного допуска.

    Бюджет 0.0 означает «эскалировать почти всё», но при rejected-гейте этого
    не происходит: решение принимает порог уверенности, reason = risk_gate.
    """
    model = _risk_model()
    model.gate = {"accepted": False, "worse": ["aurc: регрессия"]}
    cascade = RiskAwareCascadeBackend(
        _StubBackend("local", {"a": 0.9, "b": 0.4}),
        _StubBackend("remote", {"a": 0.0, "b": 0.99}),
        model,
        risk_budget=0.0,
        fallback_threshold=0.6,
    )

    result = cascade.decide("s", _questions(["a", "b"]))
    # уверенное "a" осталось локальным, несмотря на бюджет 0.0
    assert cascade.fallback.calls == [("s", ("b",))]
    assert result.usage["reason"] == "risk_gate"
    assert result.usage["escalated"] == 1

    # тот же бюджет без отклонённого сертификата — риск-правило работает
    healthy = _risk_model()
    healthy.gate = {"accepted": True, "worse": []}
    open_cascade = RiskAwareCascadeBackend(
        _StubBackend("local", {"a": 0.9, "b": 0.4}),
        _StubBackend("remote", {"a": 0.0, "b": 0.99}),
        healthy,
        risk_budget=0.0,
        fallback_threshold=0.6,
    )
    opened = open_cascade.decide("s", _questions(["a", "b"]))
    assert opened.usage["reason"] == "risk"
    # бюджет 0.0 и здоровый сертификат: уходят оба вопроса
    assert open_cascade.fallback.calls == [("s", ("a", "b"))]


def test_feedback_response_contract():
    from api.server import main

    cascade = RiskAwareCascadeBackend(
        _StubBackend("local", {}),
        _StubBackend("remote", {}),
        _risk_model(),
        risk_budget=0.5,
        few_shot_n=2,
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
                            "question": "b",
                            "decision": {"type": "noul", "noul": 0.55, "confidence": 0.1},
                            "correct": True,
                        },
                        {
                            "question": "b",
                            "decision": {"type": "noul", "noul": 0.6, "confidence": 0.2},
                            "correct": False,
                        },
                    ]
                },
            )
            assert response.status_code == 200
            body = response.json()
            row = body["questions"][0]
            assert set(row) == {"question", "accepted", "examples", "reference", "retrained"}
            assert row["accepted"] == 2 and row["examples"] == 2
            assert row["retrained"] is True
            assert isinstance(row["reference"], int)
    finally:
        main.backend = original


def test_invalid_env_values_are_rejected(monkeypatch, tmp_path):
    from api.server.main import create_cascade_backend
    from api.server.service import DecisionService

    model_path = tmp_path / "risk.json"
    _risk_model().save(model_path)
    monkeypatch.setenv("DECISION_CASCADE_FALLBACK_URL", "http://fallback")
    monkeypatch.setenv("DECISION_RISK_MODEL_PATH", str(model_path))

    monkeypatch.setenv("DECISION_CASCADE_RISK_BUDGET", "1.5")
    with pytest.raises(ValueError):
        create_cascade_backend(DecisionService(seed=0))

    monkeypatch.setenv("DECISION_CASCADE_RISK_BUDGET", "0.05")
    monkeypatch.setenv("DECISION_RISK_GATE_DELTA", "not-a-number")
    with pytest.raises(ValueError):
        create_cascade_backend(DecisionService(seed=0))

    monkeypatch.setenv("DECISION_RISK_GATE_DELTA", "0.0")
    monkeypatch.setenv("DECISION_RISK_FEW_SHOT_N", "not-a-number")
    with pytest.raises(ValueError):
        create_cascade_backend(DecisionService(seed=0))


def test_env_few_shot_below_two_disables_feedback(monkeypatch, tmp_path):
    from api.server.main import _engine_info, create_cascade_backend
    from api.server.service import DecisionService

    model_path = tmp_path / "risk.json"
    _risk_model().save(model_path)
    monkeypatch.setenv("DECISION_CASCADE_FALLBACK_URL", "http://fallback")
    monkeypatch.setenv("DECISION_CASCADE_RISK_BUDGET", "0.05")
    monkeypatch.setenv("DECISION_RISK_MODEL_PATH", str(model_path))
    monkeypatch.setenv("DECISION_RISK_FEW_SHOT_N", "1")

    backend = create_cascade_backend(DecisionService(seed=0))
    assert backend.few_shot_n is None
    assert not backend.capabilities.accepts_feedback
    info = _engine_info(backend)
    assert "risk_few_shot_n" not in info


def test_model_file_schema_is_stable(tmp_path):
    """Файл risk.json — часть контракта: поля не переименовываются в 1.x."""
    model = _risk_model()
    model.gate = {"accepted": True, "worse": []}
    payload = json.loads(model.save(tmp_path / "risk.json").read_text())
    assert set(payload) >= {"weights", "bias", "feature_names", "gate"}
    assert payload["feature_names"] == ["confidence", "entropy", "margin"]
