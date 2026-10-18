"""Тесты risk-aware каскада: эскалация по P(ошибка), причина usage, конфиг API."""
import json

import pytest
from fastapi.testclient import TestClient

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


@pytest.fixture(scope="module")
def client():
    from api.server.main import app

    with TestClient(app) as c:
        yield c


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


# --- Few-shot: транспорт примеров (``POST /v1/feedback``) ---------------------


def test_accepts_feedback_follows_few_shot_threshold():
    disabled = RiskAwareCascadeBackend(
        _StubBackend("local", {}), _StubBackend("remote", {}), _risk_model()
    )
    assert not disabled.capabilities.accepts_feedback

    enabled = RiskAwareCascadeBackend(
        _StubBackend("local", {}),
        _StubBackend("remote", {}),
        _risk_model(),
        few_shot_n=2,
    )
    assert enabled.capabilities.accepts_feedback


def test_example_counts_accumulate_by_question():
    cascade = RiskAwareCascadeBackend(
        _StubBackend("local", {}),
        _StubBackend("remote", {}),
        _risk_model(),
        few_shot_n=3,
    )
    assert cascade.example_counts == {}

    cascade.add_examples("b", [_noul(0.55), _noul(0.56)], [True, True])
    assert cascade.example_counts == {"b": 2}


def test_reference_rows_default_to_other_questions():
    from src.risk import FEATURE_NAMES

    cascade = RiskAwareCascadeBackend(
        _StubBackend("local", {}),
        _StubBackend("remote", {}),
        _risk_model(),
        few_shot_n=2,
    )
    cascade.add_examples("a", [_noul(0.9), _noul(0.1)], [True, False])

    # для "b" чужие строки — примеры "a" (та же схема, что у ``others`` офлайна)
    features, labels = cascade.reference_rows("b")
    assert features.shape == (2, len(FEATURE_NAMES))
    assert labels == [1, 0]
    # у "a" своих чужих строк нет — переобучение идёт только на своих
    assert cascade.reference_rows("a") == (None, None)


def test_explicit_reference_wins_over_accumulated():
    import numpy as np

    cascade = RiskAwareCascadeBackend(
        _StubBackend("local", {}),
        _StubBackend("remote", {}),
        _risk_model(),
        few_shot_n=2,
        reference=(np.zeros((2, 6)), [0, 1]),
    )
    cascade.add_examples("a", [_noul(0.9)], [True])

    features, labels = cascade.reference_rows("b")
    assert labels == [0, 1]
    assert not features.any()


def test_add_examples_retrains_on_reference_plus_own():
    import numpy as np

    from src.risk import features_matrix

    cascade = RiskAwareCascadeBackend(
        _StubBackend("local", {}),
        _StubBackend("remote", {}),
        _risk_model(),
        risk_budget=0.5,
        few_shot_n=2,
    )
    own = [_noul(0.55), _noul(0.60)]
    cascade.add_examples("a", [_noul(0.9)], [True])

    model = cascade.add_examples("b", own, [True, False])

    expected = _risk_model().retrained_with(
        features_matrix(own),
        np.array([1, 0]),
        reference_features=features_matrix([_noul(0.9)]),
        reference_labels=np.array([1]),
    )
    assert model.weights == pytest.approx(expected.weights)
    assert model.bias == pytest.approx(expected.bias)


def test_add_examples_is_thread_safe():
    from concurrent.futures import ThreadPoolExecutor

    cascade = RiskAwareCascadeBackend(
        _StubBackend("local", {}),
        _StubBackend("remote", {}),
        _risk_model(),
        few_shot_n=5,
    )

    def feed(index: int) -> None:
        cascade.add_examples("b", [_noul(0.5 + 0.001 * (index % 10))], [index % 2 == 0])

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(feed, range(40)))

    assert cascade.example_counts == {"b": 40}
    assert len(cascade._few_shot_models["b"].weights) == 3


def _feedback_payload(question: str, noul: float, correct: bool) -> dict:
    return {
        "examples": [
            {
                "question": question,
                "decision": {
                    "type": "noul",
                    "noul": noul,
                    "confidence": abs(2 * noul - 1),
                },
                "correct": correct,
            }
        ]
    }


def test_feedback_endpoint_rejected_without_feedback_backend(client):
    resp = client.post(
        "/v1/feedback", json=_feedback_payload("b", 0.55, True)
    )

    assert resp.status_code == 409
    assert resp.json()["error"]["code"] == "FEEDBACK_UNSUPPORTED"
    assert client.get("/info").json()["accepts_feedback"] is False


def test_feedback_endpoint_rejects_malformed_decision(client):
    resp = client.post("/v1/feedback", json=_feedback_payload("b", 1.5, True))
    assert resp.status_code == 422  # вероятность вне [0, 1] не копится в модель

    resp = client.post(
        "/v1/feedback",
        json={
            "examples": [
                {
                    "question": "b",
                    "decision": {
                        "type": "choice",
                        "choice": "вне_списка",
                        "probabilities": {"a": 0.7, "b": 0.3},
                        "confidence": 0.7,
                    },
                    "correct": True,
                }
            ]
        },
    )
    assert resp.status_code == 422


def test_feedback_endpoint_feeds_few_shot(monkeypatch):
    from api.server import main

    cascade = RiskAwareCascadeBackend(
        _StubBackend("local", {}),
        _StubBackend("remote", {}),
        _risk_model(),
        risk_budget=0.5,
        few_shot_n=3,
    )
    monkeypatch.setattr(main, "backend", cascade)

    with TestClient(main.app) as tc:
        first = tc.post(
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
                        "decision": {"type": "noul", "noul": 0.60, "confidence": 0.2},
                        "correct": False,
                    },
                    {
                        "question": "a",
                        "decision": {"type": "noul", "noul": 0.90, "confidence": 0.8},
                        "correct": True,
                    },
                ]
            },
        )
        assert first.status_code == 200
        body = first.json()
        assert body["accepted"] == 3
        # ``reference`` — чужие строки, реально взятые в обучение этого вопроса:
        # "b" обработан первым, примеров "a" ещё нет, а вот "a" уже видит два
        # примера "b"
        by_question = {row["question"]: row for row in body["questions"]}
        assert by_question["b"] == {
            "question": "b",
            "accepted": 2,
            "examples": 2,
            "reference": 0,
            "retrained": False,
        }
        assert by_question["a"] == {
            "question": "a",
            "accepted": 1,
            "examples": 1,
            "reference": 2,  # чужие строки — примеры вопроса "b"
            "retrained": False,
        }
        assert cascade.few_shot_questions == []

        # порог достигнут для "b" — модель переобучена на чужих + своих строках
        second = tc.post("/v1/feedback", json=_feedback_payload("b", 0.65, True))
        row = second.json()["questions"][0]
        assert row["examples"] == 3 and row["retrained"] is True
        assert row["reference"] == 1  # чужие строки — пример вопроса "a"
        assert cascade.few_shot_questions == ["b"]
        assert tc.get("/info").json()["accepts_feedback"] is True
