"""Тесты бенчмарка против конкурентов: mock-транспорт, оценка, каскад, расход."""
import pytest

from eval.competitors.evaluate import _flatten_usage, evaluate_backend
from eval.datasets.text_dataset import QUESTIONS, generate_examples, spec_from_api
from src.backends.base import BackendResult, Capabilities
from src.backends.protocol import answer_to_result
from src.sampler.questions import ChoiceSpec, NoulSpec, ScoreSpec


def _specs():
    return {name: spec_from_api(data) for name, data in QUESTIONS.items()}


def _mock_backend(seed: int = 0):
    from eval.competitors.mocks import mock_systemone_transport
    from src.backends.http import HttpSystemOneBackend

    return HttpSystemOneBackend(
        "http://mock", model="mock", transport=mock_systemone_transport(seed=seed)
    )


def test_mock_transport_backend_speaks_protocol():
    backend = _mock_backend()
    try:
        result = backend.decide("state text", _specs())
    finally:
        backend.close()
    assert set(result.decisions) == set(QUESTIONS)
    assert result.usage["mock"] is True


def test_evaluate_mock_backend_structure():
    specs = _specs()
    examples = generate_examples(n=8, seed=0)
    backend = _mock_backend()
    try:
        report = evaluate_backend(
            specs, examples, backend.decide, name="mock", warmup=1
        )
    finally:
        backend.close()

    assert report["name"] == "mock"
    assert report["n"] == 8
    assert 0.0 <= report["overall_accuracy"] <= 1.0
    assert set(report["questions"]) == set(specs)
    assert report["latency_ms"]["count"] == 8
    assert report["usage_totals"]["tokens"] > 0


def test_evaluate_local_backend_accuracy_range():
    from api.server.service import DecisionService
    from src.backends.local import LocalHeadsBackend

    specs = _specs()
    examples = generate_examples(n=8, seed=1)
    backend = LocalHeadsBackend(DecisionService(seed=0))
    report = evaluate_backend(specs, examples, backend.decide, name="local", warmup=1)

    assert 0.0 <= report["overall_accuracy"] <= 1.0
    # Необученные головы не должны быть идеальны на 6 вопросах.
    assert report["overall_accuracy"] < 1.0


class _EscalatingBackend:
    """Всегда возвращает уверенные полно-распределённые ответы + счётчик эскалаций."""

    name = "escalating"
    capabilities = Capabilities()

    def decide(self, state, questions, **kwargs):
        decisions = {}
        for name, spec in questions.items():
            if isinstance(spec, NoulSpec):
                decisions[name] = answer_to_result(
                    {"type": "noul", "probability": 0.5, "confidence": 0.9}
                )
            elif isinstance(spec, ChoiceSpec):
                probs = {
                    opt: (1.0 if opt == spec.options[0] else 0.0)
                    for opt in spec.options
                }
                decisions[name] = answer_to_result(
                    {
                        "type": "choice",
                        "choice": spec.options[0],
                        "confidence": 0.9,
                        "probabilities": probs,
                    }
                )
            else:
                probs = {
                    level: (1.0 if i == 0 else 0.0)
                    for i, level in enumerate(spec.levels)
                }
                decisions[name] = answer_to_result(
                    {
                        "type": "score",
                        "score": 0.0,
                        "confidence": 0.9,
                        "probabilities": probs,
                        "legend": list(spec.levels),
                    }
                )
        return BackendResult(decisions=decisions, usage={"escalated": 1})


def test_evaluate_counts_escalations():
    specs = _specs()
    examples = generate_examples(n=10, seed=2)
    report = evaluate_backend(
        specs, examples, _EscalatingBackend().decide, name="escalating", warmup=0
    )

    assert report["escalated_decisions"] == 10
    assert report["requests_with_escalation"] == 10
    assert report["escalation_rate"] == 10 / (10 * len(specs))
    assert 0.0 <= report["escalation_rate"] <= 1.0


def test_flatten_usage_sums_nested_and_ignores_bool():
    usages = [
        {"tokens": 3, "fallback": {"tokens": 5, "calls": 1}, "mock": True},
        {"tokens": 2, "fallback": {"calls": 2}, "ok": False},
    ]
    totals = _flatten_usage(usages)
    assert totals["tokens"] == 5
    assert totals["fallback.tokens"] == 5
    assert totals["fallback.calls"] == 3
    assert "mock" not in totals
    assert "ok" not in totals


def test_score_question_uses_ordinal_index_for_truth():
    # Внутренняя проверка: правда для score — порядковый индекс уровня.
    from eval.competitors.evaluate import _truth_index

    spec = _specs()["накал"]
    assert isinstance(spec, ScoreSpec)
    assert _truth_index(spec, 3) == 3
    assert _truth_index(spec, 0) == 0


def test_choice_truth_index_by_option():
    from eval.competitors.evaluate import _truth_index

    spec = _specs()["команда"]
    assert isinstance(spec, ChoiceSpec)
    assert _truth_index(spec, spec.options[2]) == 2


def test_noul_metrics_use_probabilities():
    # evaluate_backend использует noul-вероятности, а не argmax.
    from api.server.service import DecisionService
    from eval.competitors.evaluate import evaluate_backend as ev
    from src.backends.local import LocalHeadsBackend

    specs = {"u": NoulSpec("urgent?")}
    raw = generate_examples(n=12, seed=3)
    examples = [
        {"state": ex["state"], "labels": {"u": ex["labels"]["срочность"]}}
        for ex in raw
    ]
    backend = LocalHeadsBackend(DecisionService(seed=0))
    report = ev(specs, examples, backend.decide, name="local", warmup=0)
    assert report["questions"]["u"]["type"] == "noul"
    assert "brier" in report["questions"]["u"]
    assert 0.0 <= report["questions"]["u"]["brier"] <= 1.0


def test_choice_probabilities_feed_brier_and_ece():
    # Оценка берёт вероятности распределения, а не argmax: one-hot на верной
    # опции даёт нулевые Brier/ECE.
    from eval.competitors.evaluate import evaluate_backend as ev
    from src.sampler.questions import ChoiceSpec

    spec = ChoiceSpec("who?", ("a", "b"))
    examples = [{"state": f"s{i}", "labels": {"c": "a"}} for i in range(5)]
    report = ev({"c": spec}, examples, _EscalatingBackend().decide, name="x", warmup=0)

    assert report["questions"]["c"]["accuracy"] == 1.0
    assert report["questions"]["c"]["brier"] == pytest.approx(0.0, abs=1e-9)
    assert report["questions"]["c"]["ece"] == pytest.approx(0.0, abs=1e-9)


def test_run_benefit_policy_writes_policy_and_cascade(tmp_path):
    import json

    from eval.competitors.run import main

    out = tmp_path / "benefit.json"
    main(
        [
            "--specs",
            "support",
            "--limit",
            "8",
            "--mock",
            "--policy",
            "benefit",
            "--remote-accuracy",
            "0.9",
            "--out",
            str(out),
        ]
    )

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["policy"] == "benefit"
    assert payload["escalation_cost"] == pytest.approx(0.05)
    names = [report["name"] for report in payload["reports"]]
    assert any(name.startswith("benefit-cascade[") for name in names)


def test_run_benefit_policy_requires_remote_accuracy(tmp_path):
    from eval.competitors.run import main

    with pytest.raises(SystemExit, match="remote-accuracy"):
        main(
            [
                "--specs",
                "support",
                "--limit",
                "4",
                "--mock",
                "--policy",
                "benefit",
                "--out",
                str(tmp_path / "x.json"),
            ]
        )
