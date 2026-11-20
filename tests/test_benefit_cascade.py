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


def test_benefit_feedback_updates_q_only_for_escalated_answers():
    primary = _StubBackend("local", {"a": 0.95, "b": 0.40})
    fallback = _StubBackend("remote", {"a": 0.0, "b": 0.99})
    cascade = BenefitAwareCascadeBackend(primary, fallback, 0.9, escalation_cost=0.05)

    assert cascade.capabilities.accepts_feedback is True
    result = cascade.decide("s", _questions(["a", "b"]))
    assert cascade.example_counts == {}

    remote_decision = result.decisions["b"]
    cascade.add_examples("b", [remote_decision], [True])

    assert cascade.example_counts == {"b": 1}
    prior, weight = 0.9, cascade.feedback_shrinkage
    assert cascade.remote_accuracy_for("b") == pytest.approx(
        (prior * weight + 1) / (weight + 1)
    )
    # ответ локальной головы качество внешней модели не характеризует
    cascade.add_examples("a", [result.decisions["a"]], [False])
    assert cascade.example_counts == {"b": 1}
    # повторный фидбэк по тому же выданному решению не считается дважды
    cascade.add_examples("b", [remote_decision], [True])
    assert cascade.example_counts == {"b": 1}
    # у вопроса без наблюдений q остаётся априорной
    assert cascade.remote_accuracy_for("c") == pytest.approx(0.9)


def test_benefit_feedback_can_stop_escalation():
    # q падает ниже локальной уверенности -> политика перестаёт платить за вызов
    primary = _StubBackend("local", {"a": 0.5})
    fallback = _StubBackend("remote", {"a": 0.99})
    cascade = BenefitAwareCascadeBackend(
        primary, fallback, 0.9, escalation_cost=0.0, feedback_shrinkage=1.0
    )
    assert cascade.decide("s", _questions(["a"])).usage["escalated"] == 1

    cascade.add_examples("a", [fallback.decide("s", _questions(["a"])).decisions["a"]], [False])

    assert cascade.remote_accuracy_for("a") == pytest.approx(0.45)
    assert cascade.decide("s", _questions(["a"])).usage["escalated"] == 0


def test_feedback_endpoint_reports_remote_accuracy(monkeypatch):
    from fastapi.testclient import TestClient

    from api.server import main

    primary = _StubBackend("local", {"a": 0.40})
    fallback = _StubBackend("remote", {"a": 0.99})
    cascade = BenefitAwareCascadeBackend(primary, fallback, 0.9, escalation_cost=0.05)
    result = cascade.decide("s", _questions(["a"]))
    monkeypatch.setattr(main, "backend", cascade)

    payload = {
        "examples": [
            {
                "question": "a",
                "decision": {"type": "noul", "noul": 0.99, "confidence": 0.99},
                "correct": True,
            }
        ]
    }
    with TestClient(main.app) as tc:
        response = tc.post("/v1/feedback", json=payload)
        # повтор того же фидбэка не должен раздувать счётчик принятых ответов
        again = tc.post("/v1/feedback", json=payload)

    assert response.status_code == 200
    row = response.json()["questions"][0]
    assert row["question"] == "a"
    assert row["examples"] == 1
    assert row["retrained"] is False
    assert row["remote_accuracy"] == pytest.approx(cascade.remote_accuracy_for("a"))
    assert again.json()["questions"][0]["examples"] == 1
    assert result.usage["escalated"] == 1


# --- Журнал фидбэка: счётчики переживают рестарт ------------------------------


def _benefit_cascade(path):
    primary = _StubBackend("local", {"a": 0.95, "b": 0.40})
    fallback = _StubBackend("remote", {"a": 0.0, "b": 0.99})
    return BenefitAwareCascadeBackend(
        primary, fallback, 0.9, escalation_cost=0.05, feedback_path=path
    )


def test_benefit_feedback_survives_restart(tmp_path):
    path = tmp_path / "feedback.jsonl"
    first = _benefit_cascade(path)
    remote_decision = first.decide("s", _questions(["a", "b"])).decisions["b"]
    first.add_examples("b", [remote_decision], [True])
    q_before = first.remote_accuracy_for("b")

    second = _benefit_cascade(path)

    assert second.example_counts == {"b": 1}
    assert second.restored_examples == 1
    assert second.remote_accuracy_for("b") == pytest.approx(q_before)


def test_benefit_feedback_duplicates_not_persisted(tmp_path):
    path = tmp_path / "feedback.jsonl"
    first = _benefit_cascade(path)
    remote_decision = first.decide("s", _questions(["b"])).decisions["b"]
    first.add_examples("b", [remote_decision], [True])
    # повтор по тому же выданному решению не учитывается и в журнал не идёт
    first.add_examples("b", [remote_decision], [True])

    assert first.example_counts == {"b": 1}
    assert _benefit_cascade(path).example_counts == {"b": 1}


def test_benefit_feedback_endpoint_persists(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from api.server import main

    path = tmp_path / "feedback.jsonl"
    cascade = _benefit_cascade(path)
    cascade.decide("s", _questions(["b"]))
    monkeypatch.setattr(main, "backend", cascade)

    payload = {
        "examples": [
            {
                "question": "b",
                "decision": {"type": "noul", "noul": 0.99, "confidence": 0.99},
                "correct": True,
            }
        ]
    }
    with TestClient(main.app) as tc:
        response = tc.post("/v1/feedback", json=payload)

    assert response.status_code == 200
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert _benefit_cascade(path).example_counts == {"b": 1}


# --- Исследовательские эскалации и IPS-оценка q -------------------------------


def test_benefit_exploration_off_by_default():
    cascade = BenefitAwareCascadeBackend(
        _StubBackend("local", {"a": 0.95}),
        _StubBackend("remote", {"a": 0.99}),
        0.9,
    )
    assert cascade.explore == 0.0
    info = cascade.benefit_info()
    assert info["explore_rate"] == 0.0
    assert info["explore_calls"] == 0


def test_benefit_exploration_escalates_and_weights_feedback(tmp_path):
    path = tmp_path / "feedback.jsonl"

    class _AlwaysExplore:
        def random(self):
            return 0.0

    primary = _StubBackend("local", {"a": 0.95})
    fallback = _StubBackend("remote", {"a": 0.99})
    cascade = BenefitAwareCascadeBackend(
        primary,
        fallback,
        0.9,
        escalation_cost=0.05,
        explore=0.5,
        feedback_path=path,
    )
    # политика оставила бы «a» локально: 0.95 > 0.9 - 0.025, но жребий всегда «да»
    cascade._rng = _AlwaysExplore()

    result = cascade.decide("s", _questions(["a"]))

    assert result.usage["escalated"] == 1
    assert result.usage["reason"] == "benefit"
    assert cascade.explore_calls == 1
    decision = result.decisions["a"]
    cascade.add_examples("a", [decision], [True])

    weight = 2.0  # 1 / explore
    assert cascade.example_counts == {"a": 1}
    assert cascade.remote_accuracy_for("a") == pytest.approx(
        (0.9 * 50 + weight) / (50 + weight)
    )
    row = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert row["propensity"] == pytest.approx(0.5)

    # рестарт поднимает и счётчики, и IPS-веса: q не откатывается к априорной
    again = BenefitAwareCascadeBackend(
        _StubBackend("local", {"a": 0.95}),
        _StubBackend("remote", {"a": 0.99}),
        0.9,
        escalation_cost=0.05,
        explore=0.5,
        feedback_path=path,
    )
    assert again.example_counts == {"a": 1}
    assert again.remote_accuracy_for("a") == pytest.approx(
        cascade.remote_accuracy_for("a")
    )


def test_benefit_exploration_is_seeded():
    def run():
        primary = _StubBackend("local", {"a": 0.95})
        fallback = _StubBackend("remote", {"a": 0.99})
        cascade = BenefitAwareCascadeBackend(
            primary, fallback, 0.9, explore=0.5, explore_seed=42
        )
        for _ in range(100):
            cascade.decide("s", _questions(["a"]))
        return fallback.calls, cascade.explore_calls

    first = run()
    second = run()

    assert first == second
    assert 0 < first[1] < 100


def test_benefit_explore_validated():
    with pytest.raises(ValueError, match="explore"):
        BenefitAwareCascadeBackend(
            _StubBackend("local", {}), _StubBackend("remote", {}), 0.9, explore=1.5
        )


def test_create_cascade_backend_reads_explore(monkeypatch):
    from api.server.main import _engine_info, create_cascade_backend
    from api.server.service import DecisionService

    monkeypatch.setenv("DECISION_CASCADE_FALLBACK_URL", "http://fallback")
    monkeypatch.setenv("DECISION_CASCADE_POLICY", "benefit")
    monkeypatch.setenv("DECISION_CASCADE_REMOTE_ACCURACY", "0.86")
    monkeypatch.setenv("DECISION_CASCADE_EXPLORE", "0.05")
    monkeypatch.setenv("DECISION_CASCADE_EXPLORE_SEED", "7")
    monkeypatch.delenv("DECISION_CASCADE_RISK_BUDGET", raising=False)

    backend = create_cascade_backend(DecisionService(seed=0))

    assert backend.explore == pytest.approx(0.05)
    info = _engine_info(backend)
    assert info["explore_rate"] == pytest.approx(0.05)
    assert info["explore_calls"] == 0
