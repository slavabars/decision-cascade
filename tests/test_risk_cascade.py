"""Тесты risk-aware каскада: эскалация по P(ошибка), причина usage, конфиг API."""
import json

import pytest

from src.backends.base import BackendResult, Capabilities
from src.backends.cascade import RiskAwareCascadeBackend
from src.risk import LEGACY_FEATURE_NAMES, ErrorRiskModel
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


def _noul(confidence):
    return NoulResult(noul=confidence, confidence=confidence)


def _risk_model():
    # z = -20*confidence + 14: уверенно -> P(ошибка) ~0, у границы 0.5 -> ~0.95
    return ErrorRiskModel(
        weights=np_error_weights(),
        bias=14.0,
        feature_names=LEGACY_FEATURE_NAMES,
    )


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


def test_risk_cascade_falls_back_to_confidence_when_gate_rejected():
    """Отказ гейта: риск-правило не применяется, работает порог уверенности."""
    model = _risk_model()
    # при budget=0.99 риск-правило ничего не эскалирует (p(error) < 0.99)
    assert not model.should_escalate(
        NoulResult(noul=0.55, confidence=0.55), 0.99
    )
    model.gate = {"accepted": False, "worse": ["loo_aurc: 0.61 > 0.61±0.0005"]}
    primary = _StubBackend("local", {"a": 0.9, "b": 0.55})
    fallback = _StubBackend("remote", {"a": 0.0, "b": 0.99})

    cascade = RiskAwareCascadeBackend(primary, fallback, model, risk_budget=0.99)
    result = cascade.decide("s", _questions(["a", "b"]))

    assert cascade.gate_rejected is True
    assert fallback.calls == [("s", ("b",))]
    assert result.usage["escalated"] == 1
    assert result.usage["reason"] == "risk_gate"


def test_risk_cascade_keeps_risk_rule_when_gate_accepted():
    model = _risk_model()
    model.gate = {"accepted": True, "worse": []}
    primary = _StubBackend("local", {"a": 0.9, "b": 0.55})
    fallback = _StubBackend("remote", {"a": 0.0, "b": 0.99})

    cascade = RiskAwareCascadeBackend(primary, fallback, model, risk_budget=0.5)
    result = cascade.decide("s", _questions(["a", "b"]))

    assert cascade.gate_rejected is False
    assert fallback.calls == [("s", ("b",))]
    assert result.usage["reason"] == "risk"


def test_risk_cascade_without_gate_certificate_unchanged():
    """Замеры не проводились — отказывать нечего, режим работает как прежде."""
    model = _risk_model()
    assert model.gate is None
    assert not model.gate_rejected
    primary = _StubBackend("local", {"a": 0.9, "b": 0.55})
    fallback = _StubBackend("remote", {"a": 0.0, "b": 0.99})

    cascade = RiskAwareCascadeBackend(primary, fallback, model, risk_budget=0.5)
    cascade.decide("s", _questions(["a", "b"]))

    assert fallback.calls == [("s", ("b",))]
    assert cascade.escalation_reason == "risk"


def test_risk_cascade_fallback_threshold_validated():
    from api.server.service import DecisionService
    from src.backends.http import HttpSystemOneBackend, parse_modalities
    from src.backends.local import LocalHeadsBackend

    primary = LocalHeadsBackend(DecisionService(seed=0))
    fallback = HttpSystemOneBackend(
        "http://x", model="remote", modalities=parse_modalities("text")
    )
    with pytest.raises(ValueError):
        RiskAwareCascadeBackend(
            primary, fallback, _risk_model(), fallback_threshold=1.5
        )


def test_gate_certificate_survives_roundtrip(tmp_path):
    model = _risk_model()
    model.gate = {"accepted": False, "worse": ["aurc: 0.61 > 0.61±0.0005"],
                  "tol": 0.0005}
    path = model.save(tmp_path / "risk.json")

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["gate"]["accepted"] is False

    loaded = ErrorRiskModel.load(path)
    assert loaded.gate_rejected
    assert loaded.gate["worse"] == ["aurc: 0.61 > 0.61±0.0005"]
    # старый файл без сертификата — не отказ
    legacy = ErrorRiskModel.from_dict(
        {"weights": [0.0, 0.0, 0.0], "bias": 0.0}
    )
    assert legacy.gate is None and not legacy.gate_rejected


def test_create_cascade_backend_applies_rejected_gate(monkeypatch, tmp_path):
    from api.server.main import _engine_info, create_cascade_backend
    from api.server.service import DecisionService

    model_path = tmp_path / "risk.json"
    model = _risk_model()
    model.gate = {"accepted": False, "worse": ["aurc: 0.61 > 0.61±0.0005"]}
    model.save(model_path)
    monkeypatch.setenv("DECISION_CASCADE_FALLBACK_URL", "http://fallback")
    monkeypatch.setenv("DECISION_CASCADE_RISK_BUDGET", "0.12")
    monkeypatch.setenv("DECISION_RISK_MODEL_PATH", str(model_path))
    monkeypatch.setenv("DECISION_CASCADE_THRESHOLD", "0.75")

    backend = create_cascade_backend(DecisionService(seed=0))

    assert isinstance(backend, RiskAwareCascadeBackend)
    assert backend.gate_rejected
    assert backend.escalation_reason == "risk_gate"
    assert backend.fallback_threshold == pytest.approx(0.75)
    info = _engine_info(backend)
    assert info["risk_gate"] == "rejected"
    assert info["risk_budget"] == pytest.approx(0.12)


def test_risk_cascade_ignores_few_shot_without_threshold():
    from src.risk import RISK_FEW_SHOT_MIN_N

    assert RISK_FEW_SHOT_MIN_N == 10
    cascade = RiskAwareCascadeBackend(
        _StubBackend("local", {"b": 0.55}),
        _StubBackend("remote", {"b": 0.99}),
        _risk_model(),
        risk_budget=0.5,
    )
    assert cascade.few_shot_n is None
    assert cascade.add_examples("b", [_noul(0.55)], [True]) is None
    assert cascade.few_shot_questions == []


def test_risk_cascade_retrains_after_few_shot_threshold():
    primary = _StubBackend("local", {"a": 0.9, "b": 0.55})
    fallback = _StubBackend("remote", {"a": 0.0, "b": 0.99})
    cascade = RiskAwareCascadeBackend(
        primary, fallback, _risk_model(), risk_budget=0.5, few_shot_n=3
    )

    # до порога свои примеры только копятся — базовая модель эскалирует "b"
    assert cascade.add_examples("b", [_noul(0.55), _noul(0.56)], [True, True]) is None
    assert cascade.few_shot_questions == []
    cascade.decide("s", _questions(["a", "b"]))
    assert fallback.calls == [("s", ("b",))]

    # порог достигнут: "b" переобучен на своих (все верные) — риск около нуля
    model = cascade.add_examples("b", [_noul(0.55)], [True])
    assert model is not None
    assert cascade.few_shot_questions == ["b"]
    assert cascade._model_for("b") is model
    assert cascade._model_for("a") is cascade.risk_model

    fallback.calls.clear()
    result = cascade.decide("s", _questions(["a", "b"]))
    assert fallback.calls == []
    assert result.usage["escalated"] == 0


def test_add_examples_rejects_length_mismatch():
    cascade = RiskAwareCascadeBackend(
        _StubBackend("local", {}),
        _StubBackend("remote", {}),
        _risk_model(),
        few_shot_n=2,
    )
    with pytest.raises(ValueError, match="same length"):
        cascade.add_examples("b", [_noul(0.5)], [True, False])


def test_risk_cascade_few_shot_n_validated():
    with pytest.raises(ValueError, match="few_shot_n"):
        RiskAwareCascadeBackend(
            _StubBackend("local", {}),
            _StubBackend("remote", {}),
            _risk_model(),
            few_shot_n=1,
        )


def test_create_cascade_backend_reads_few_shot_and_gate_delta(monkeypatch, tmp_path):
    from api.server.main import _engine_info, create_cascade_backend
    from api.server.service import DecisionService
    from src.risk import RISK_FEW_SHOT_MIN_N

    model_path = tmp_path / "risk.json"
    _risk_model().save(model_path)
    monkeypatch.setenv("DECISION_CASCADE_FALLBACK_URL", "http://fallback")
    monkeypatch.setenv("DECISION_CASCADE_RISK_BUDGET", "0.12")
    monkeypatch.setenv("DECISION_RISK_MODEL_PATH", str(model_path))
    monkeypatch.delenv("DECISION_RISK_FEW_SHOT_N", raising=False)
    monkeypatch.setenv("DECISION_RISK_GATE_DELTA", "0.02")

    backend = create_cascade_backend(DecisionService(seed=0))

    assert backend.few_shot_n == RISK_FEW_SHOT_MIN_N  # дефолт Шага 6
    assert backend.gate_delta == pytest.approx(0.02)
    info = _engine_info(backend)
    assert info["risk_few_shot_n"] == RISK_FEW_SHOT_MIN_N
    assert info["risk_few_shot"] == []
    assert info["risk_gate_delta"] == pytest.approx(0.02)


def test_create_cascade_backend_few_shot_can_be_disabled(monkeypatch, tmp_path):
    from api.server.main import _engine_info, create_cascade_backend
    from api.server.service import DecisionService

    model_path = tmp_path / "risk.json"
    _risk_model().save(model_path)
    monkeypatch.setenv("DECISION_CASCADE_FALLBACK_URL", "http://fallback")
    monkeypatch.setenv("DECISION_CASCADE_RISK_BUDGET", "0.12")
    monkeypatch.setenv("DECISION_RISK_MODEL_PATH", str(model_path))
    monkeypatch.setenv("DECISION_RISK_FEW_SHOT_N", "0")

    backend = create_cascade_backend(DecisionService(seed=0))

    assert backend.few_shot_n is None
    assert "risk_few_shot_n" not in _engine_info(backend)


def test_gate_delta_relaxes_rejected_certificate(monkeypatch, tmp_path):
    from api.server.main import create_cascade_backend
    from api.server.service import DecisionService

    model = _risk_model()
    model.gate = {"accepted": False, "worse": ["aurc: регрессия"], "excess": {"aurc": 0.004}}
    model_path = tmp_path / "risk.json"
    model.save(model_path)
    monkeypatch.setenv("DECISION_CASCADE_FALLBACK_URL", "http://fallback")
    monkeypatch.setenv("DECISION_CASCADE_RISK_BUDGET", "0.12")
    monkeypatch.setenv("DECISION_RISK_MODEL_PATH", str(model_path))

    monkeypatch.setenv("DECISION_RISK_GATE_DELTA", "0.0")
    assert create_cascade_backend(DecisionService(seed=0)).gate_rejected

    # допуск шире регрессии — сертификат принимается, риск-правило применяется
    monkeypatch.setenv("DECISION_RISK_GATE_DELTA", "0.01")
    relaxed = create_cascade_backend(DecisionService(seed=0))
    assert not relaxed.gate_rejected
    assert relaxed.escalation_reason == "risk"
