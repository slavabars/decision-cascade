"""Тесты risk-aware каскада: эскалация по P(ошибка), причина usage, конфиг API."""
import json

import pytest

from src.backends.base import BackendResult, Capabilities
from src.backends.cascade import RiskAwareCascadeBackend
from src.risk import ErrorRiskModel
from src.sampler.questions import NoulSpec
from src.schemas import NoulResult


class _StubBackend:
    def __init__(self, name, decision_map):
        self.name = name
        self.decision_map = decision_map
        self.capabilities = Capabilities(modalities=frozenset({"text"}))
        self.calls = []

    def decide(self, state, questions, *, images=None, videos=None, media_kwargs=None):
        self.calls.append((state, tuple(questions)))
        decisions = {
            name: NoulResult(noul=self.decision_map[name], confidence=self.decision_map[name])
            for name in questions
        }
        return BackendResult(decisions=decisions, usage={"calls": 1})

    def close(self):
        pass


def _questions(names):
    return {name: NoulSpec(f"?{name}") for name in names}


def _risk_model():
    # z = -20*confidence + 14: уверенно -> P(ошибка) ~0, у границы 0.5 -> ~0.95
    return ErrorRiskModel(weights=np_error_weights(), bias=14.0)


def np_error_weights():
    import numpy as np

    return np.array([-20.0, 0.0, 0.0])


def test_risk_cascade_escalates_low_confidence():
    primary = _StubBackend("local", {"a": 0.9, "b": 0.55})
    fallback = _StubBackend("remote", {"a": 0.0, "b": 0.99})
    cascade = RiskAwareCascadeBackend(primary, fallback, _risk_model(), risk_budget=0.5)

    result = cascade.decide("s", _questions(["a", "b"]))

    assert fallback.calls == [("s", ("b",))]
    assert result.decisions["b"].confidence == 0.99
    assert result.usage["escalated"] == 1
    assert result.usage["reason"] == "risk"


def test_risk_cascade_skips_when_all_confident():
    primary = _StubBackend("local", {"a": 0.95, "b": 0.92})
    fallback = _StubBackend("remote", {"a": 0.0, "b": 0.0})
    cascade = RiskAwareCascadeBackend(primary, fallback, _risk_model(), risk_budget=0.05)

    result = cascade.decide("s", _questions(["a", "b"]))

    assert fallback.calls == []
    assert result.usage["escalated"] == 0


def test_risk_cascade_budget_bounds_validated():
    from api.server.service import DecisionService
    from src.backends.http import HttpSystemOneBackend, parse_modalities
    from src.backends.local import LocalHeadsBackend

    primary = LocalHeadsBackend(DecisionService(seed=0))
    fallback = HttpSystemOneBackend(
        "http://x", model="remote", modalities=parse_modalities("text")
    )
    with pytest.raises(ValueError):
        RiskAwareCascadeBackend(primary, fallback, _risk_model(), risk_budget=1.5)


def test_create_cascade_backend_requires_risk_model(monkeypatch, tmp_path):
    from api.server.main import create_cascade_backend
    from api.server.service import DecisionService

    monkeypatch.setenv("DECISION_CASCADE_FALLBACK_URL", "http://fallback")
    monkeypatch.setenv("DECISION_CASCADE_RISK_BUDGET", "0.05")
    monkeypatch.setenv("DECISION_RISK_MODEL_PATH", str(tmp_path / "missing.json"))

    with pytest.raises(RuntimeError, match="risk model"):
        create_cascade_backend(DecisionService(seed=0))


def test_create_cascade_backend_builds_risk_aware(monkeypatch, tmp_path):
    from api.server.main import create_cascade_backend
    from api.server.service import DecisionService

    model_path = tmp_path / "risk.json"
    _risk_model().save(model_path)
    monkeypatch.setenv("DECISION_CASCADE_FALLBACK_URL", "http://fallback")
    monkeypatch.setenv("DECISION_CASCADE_RISK_BUDGET", "0.12")
    monkeypatch.setenv("DECISION_RISK_MODEL_PATH", str(model_path))

    backend = create_cascade_backend(DecisionService(seed=0))

    assert isinstance(backend, RiskAwareCascadeBackend)
    assert backend.risk_budget == pytest.approx(0.12)
    assert backend.escalation_reason == "risk"


def test_info_reports_risk_budget(monkeypatch, tmp_path):
    from api.server.main import _engine_info

    model_path = tmp_path / "risk.json"
    _risk_model().save(model_path)

    class _Engine:
        def info(self):
            return {"name": "stub", "engine": "local"}

    class _Stub:
        engine = _Engine()

    class _Backend:
        risk_budget = 0.07
        primary = _Stub()
        fallback = None

    info = _engine_info(_Backend())
    assert info["risk_budget"] == pytest.approx(0.07)
    assert info["primary"]["name"] == "stub"


def test_risk_json_roundtrip_file(tmp_path):
    model = _risk_model()
    path = model.save(tmp_path / "risk.json")
    payload = json.loads(path.read_text())
    assert payload["feature_names"] == ["confidence", "entropy", "margin"]
    assert "weights" in payload and "bias" in payload
