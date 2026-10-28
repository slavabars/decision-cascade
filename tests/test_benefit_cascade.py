"""Тесты benefit-aware каскада: порог из выгоды, per-question q, парсинг q."""
import json

import pytest

from src.backends.base import BackendResult, Capabilities
from src.backends.benefit import BenefitAwareCascadeBackend, load_remote_accuracy
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


def test_benefit_escalates_only_below_measured_threshold():
    # q = 0.9, C = 0.2, D = 1 -> порог 0.9 - 0.1 = 0.8
    primary = _StubBackend("local", {"a": 0.95, "b": 0.70})
    fallback = _StubBackend("remote", {"a": 0.0, "b": 0.99})
    cascade = BenefitAwareCascadeBackend(
        primary, fallback, 0.9, escalation_cost=0.2, error_cost=1.0
    )

    result = cascade.decide("s", _questions(["a", "b"]))

    assert fallback.calls == [("s", ("b",))]
    assert result.decisions["b"].confidence == 0.99
    assert result.usage["escalated"] == 1
    assert result.usage["reason"] == "benefit"
    assert cascade.threshold_for("a") == pytest.approx(0.8)


def test_benefit_per_question_accuracy():
    # На вопросе «b» внешняя модель слабее локальной -> эскалации нет.
    primary = _StubBackend("local", {"a": 0.6, "b": 0.6})
    fallback = _StubBackend("remote", {"a": 0.9, "b": 0.2})
    cascade = BenefitAwareCascadeBackend(
        primary,
        fallback,
        {"a": 0.9, "b": 0.2},
        escalation_cost=0.0,
    )

    result = cascade.decide("s", _questions(["a", "b"]))

    assert fallback.calls == [("s", ("a",))]
    assert result.usage["escalated"] == 1
    # средняя q по вопросам — 0.55, для неизвестного вопроса берётся она
    assert cascade.remote_accuracy_for("c") == pytest.approx(0.55)


def test_benefit_never_escalates_when_remote_is_not_better():
    primary = _StubBackend("local", {"a": 0.99})
    fallback = _StubBackend("remote", {"a": 0.0})
    cascade = BenefitAwareCascadeBackend(primary, fallback, 0.4, escalation_cost=0.05)

    result = cascade.decide("s", _questions(["a"]))

    assert fallback.calls == []
    assert result.usage["escalated"] == 0


def test_benefit_expected_gain_sign_matches_decision():
    primary = _StubBackend("local", {"a": 0.5})
    fallback = _StubBackend("remote", {"a": 0.0})
    cascade = BenefitAwareCascadeBackend(
        primary, fallback, 0.9, escalation_cost=0.1, error_cost=0.0
    )

    assert cascade.expected_gain("a", 0.5) == pytest.approx(0.3)
    assert cascade.expected_gain("a", 0.85) == pytest.approx(-0.05)
    assert cascade.threshold_for("a") == pytest.approx(0.8)


def test_benefit_batch_uses_same_rule():
    primary = _StubBackend("local", {"a": 0.95, "b": 0.40})
    fallback = _StubBackend("remote", {"a": 0.0, "b": 0.0})
    cascade = BenefitAwareCascadeBackend(primary, fallback, 0.9, escalation_cost=0.05)

    results = cascade.decide_many(["s1", "s2"], _questions(["a", "b"]))

    # эскалируется только «b» — по одному вопросу в каждом состоянии
    assert fallback.calls == [("s1", ("b",)), ("s2", ("b",))]
    assert [r.usage["escalated"] for r in results] == [1, 1]
    assert results[0].decisions["a"].confidence == pytest.approx(0.95)


def test_benefit_rejects_invalid_parameters():
    primary = _StubBackend("local", {"a": 0.5})
    fallback = _StubBackend("remote", {"a": 0.5})

    with pytest.raises(ValueError):
        BenefitAwareCascadeBackend(primary, fallback, 1.5)
    with pytest.raises(ValueError):
        BenefitAwareCascadeBackend(primary, fallback, {})
    with pytest.raises(ValueError):
        BenefitAwareCascadeBackend(primary, fallback, 0.9, escalation_cost=-0.1)
    with pytest.raises(ValueError):
        BenefitAwareCascadeBackend(primary, fallback, 0.9, error_cost=-1.0)
    with pytest.raises(ValueError):
        BenefitAwareCascadeBackend(primary, fallback, 0.0)


def test_load_remote_accuracy_accepts_number_and_json(tmp_path):
    assert load_remote_accuracy("0.86") == pytest.approx(0.86)

    path = tmp_path / "q.json"
    path.write_text(
        json.dumps({"remote": "clef", "accuracy": 0.7}), encoding="utf-8"
    )
    assert load_remote_accuracy(str(path)) == pytest.approx(0.7)

    per_question = tmp_path / "q2.json"
    per_question.write_text(
        json.dumps(
            {"accuracy": 0.6, "per_question": {"topic": 0.86, "business": 0.5}}
        ),
        encoding="utf-8",
    )
    assert load_remote_accuracy(str(per_question)) == {
        "topic": pytest.approx(0.86),
        "business": pytest.approx(0.5),
    }

    with pytest.raises(ValueError):
        load_remote_accuracy("1.7")
    with pytest.raises(ValueError):
        load_remote_accuracy(str(tmp_path / "missing.json"))


def test_create_cascade_backend_builds_benefit(monkeypatch):
    from api.server.main import create_cascade_backend
    from api.server.service import DecisionService

    monkeypatch.setenv("DECISION_CASCADE_FALLBACK_URL", "http://fallback")
    monkeypatch.setenv("DECISION_CASCADE_POLICY", "benefit")
    monkeypatch.setenv("DECISION_CASCADE_REMOTE_ACCURACY", "0.86")
    monkeypatch.setenv("DECISION_CASCADE_ESCALATION_COST", "0.1")
    monkeypatch.delenv("DECISION_CASCADE_RISK_BUDGET", raising=False)

    backend = create_cascade_backend(DecisionService(seed=0))

    assert isinstance(backend, BenefitAwareCascadeBackend)
    assert backend.remote_accuracy_for("anything") == pytest.approx(0.86)
    assert backend.escalation_cost == pytest.approx(0.1)


def test_create_cascade_backend_benefit_requires_accuracy(monkeypatch):
    from api.server.main import create_cascade_backend
    from api.server.service import DecisionService

    monkeypatch.setenv("DECISION_CASCADE_FALLBACK_URL", "http://fallback")
    monkeypatch.setenv("DECISION_CASCADE_POLICY", "benefit")
    monkeypatch.delenv("DECISION_CASCADE_REMOTE_ACCURACY", raising=False)
    monkeypatch.delenv("DECISION_CASCADE_RISK_BUDGET", raising=False)

    with pytest.raises(RuntimeError, match="DECISION_CASCADE_REMOTE_ACCURACY"):
        create_cascade_backend(DecisionService(seed=0))


def test_wilson_lower_is_conservative():
    from eval.competitors.fit_remote_accuracy import wilson_lower

    assert wilson_lower(0.4, 100) == pytest.approx(0.31, abs=0.02)
    assert wilson_lower(0.4, 400) > wilson_lower(0.4, 100)
    assert wilson_lower(0.0, 100) == 0.0
    assert wilson_lower(1.0, 100) < 1.0


def test_load_remote_accuracy_conservative_uses_lower_bound(tmp_path):
    path = tmp_path / "q.json"
    path.write_text(
        json.dumps(
            {
                "accuracy": 0.6,
                "accuracy_lower": 0.5,
                "per_question": {"topic": 0.86, "business": 0.5},
                "per_question_lower": {"topic": 0.78, "business": 0.40},
            }
        ),
        encoding="utf-8",
    )

    assert load_remote_accuracy(str(path), conservative=True) == {
        "topic": pytest.approx(0.78),
        "business": pytest.approx(0.40),
    }
    assert load_remote_accuracy(str(path))["topic"] == pytest.approx(0.86)
