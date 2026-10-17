"""Тесты честного сравнения маршрутизации (eval/risk/core)."""
import json

import numpy as np
import pytest

from eval.risk.core import (
    GATE_AB_METRICS,
    GATE_LOO_METRICS,
    GATE_SHOTS_METRICS,
    GateMetric,
    bootstrap_mean_ci,
    bootstrap_ranker_ci,
    cascade_summary,
    flatten,
    gate_risk_vs_confidence,
    is_correct,
    labels_for,
    paired_differences,
    ranker_report,
    risk_escalation,
    selective_accuracy,
    stability_report,
    threshold_escalation,
)
from src.risk import LEGACY_FEATURE_NAMES, ErrorRiskModel
from src.sampler.questions import ChoiceSpec, NoulSpec, ScoreSpec
from src.schemas import ChoiceResult, NoulResult, ScoreResult


def test_cascade_summary_ideal_fallback():
    correct = np.array([1.0, 1.0, 0.0, 0.0])
    escalate = np.array([False, False, True, False])
    summary = cascade_summary(correct, escalate)
    assert summary["escalation_rate"] == pytest.approx(0.25)
    assert summary["accepted_rate"] == pytest.approx(0.75)
    assert summary["accepted_accuracy"] == pytest.approx(2 / 3)
    assert summary["accepted_risk"] == pytest.approx(1 / 3)
    assert summary["end2end_accuracy"] == pytest.approx(0.75)


def test_cascade_summary_no_escalation():
    correct = np.array([1.0, 1.0])
    escalate = np.array([False, False])
    summary = cascade_summary(correct, escalate)
    assert summary["escalation_rate"] == 0.0
    assert summary["accepted_accuracy"] == pytest.approx(1.0)
    assert summary["end2end_accuracy"] == pytest.approx(1.0)


def test_cascade_summary_all_escalated():
    correct = np.array([1.0, 0.0])
    escalate = np.array([True, True])
    summary = cascade_summary(correct, escalate)
    assert summary["escalation_rate"] == 1.0
    assert np.isnan(summary["accepted_accuracy"])
    assert summary["end2end_accuracy"] == pytest.approx(1.0)  # fallback идеален


def test_threshold_escalation_marks_low_confidence():
    decisions = [
        NoulResult(noul=0.9, confidence=0.9),
        NoulResult(noul=0.55, confidence=0.3),
    ]
    flags = threshold_escalation(decisions, threshold=0.6)
    assert flags.tolist() == [False, True]


def test_risk_escalation_uses_model():
    model = ErrorRiskModel(
        weights=np.array([-20.0, 0.0, 0.0]),
        bias=14.0,
        feature_names=LEGACY_FEATURE_NAMES,
    )
    decisions = [
        NoulResult(noul=0.9, confidence=0.9),
        NoulResult(noul=0.55, confidence=0.3),
    ]
    flags = risk_escalation(decisions, model, budget=0.5)
    assert flags.tolist() == [False, True]


def test_ranker_report_keys_and_bounds():
    scores = np.array([0.9, 0.8, 0.3, 0.2])
    correct = np.array([1.0, 1.0, 0.0, 0.0])
    report = ranker_report(scores, correct)
    assert set(report) == {
        "aurc",
        "eaurc",
        "coverage_at_5pct_risk",
        "coverage_at_10pct_risk",
        "ece",
        "brier",
    }
    assert report["eaurc"] == pytest.approx(0.0, abs=1e-9)
    assert 0.0 <= report["coverage_at_5pct_risk"] <= 1.0
    # балл идеально упорядочен и калиброван: ошибка 1 при score<=0.3
    assert 0.0 <= report["ece"] <= 1.0
    assert 0.0 <= report["brier"] <= 1.0


def test_ranker_report_calibration_is_error_calibration():
    """ECE/Brier считаются на P(ошибка) = 1 - score, поэтому идеальный score даёт 0."""
    scores = np.array([1.0, 1.0, 0.0, 0.0])
    correct = np.array([1.0, 1.0, 0.0, 0.0])
    report = ranker_report(scores, correct)
    assert report["ece"] == pytest.approx(0.0, abs=1e-9)
    assert report["brier"] == pytest.approx(0.0, abs=1e-9)

    # score 0.5 при верных ответах: P(ошибка) предсказано 0.5, ошибок нет
    uncertain = ranker_report(np.array([0.5, 0.5]), np.array([1.0, 1.0]))
    assert uncertain["brier"] == pytest.approx(0.25)
    assert uncertain["ece"] == pytest.approx(0.5)


def test_labels_and_is_correct_noul():
    spec = NoulSpec("urgent?")
    examples = [{"labels": {"u": True}}, {"labels": {"u": False}}]
    labels = labels_for(spec, examples, "u")
    assert labels.tolist() == [1.0, 0.0]
    assert is_correct(spec, NoulResult(noul=0.9, confidence=0.8), labels[0])
    assert not is_correct(spec, NoulResult(noul=0.2, confidence=0.6), labels[0])


def test_is_correct_choice_and_score():
    choice = ChoiceSpec("which?", options=("yes", "no"))
    assert is_correct(choice, ChoiceResult(choice="yes", probabilities={"yes": 0.8, "no": 0.2}, confidence=0.8), 0)
    assert not is_correct(choice, ChoiceResult(choice="no", probabilities={"yes": 0.8, "no": 0.2}, confidence=0.8), 0)

    score = ScoreSpec("how many?", levels=("low", "high"))
    result = ScoreResult(
        score=1.0,
        probabilities={"low": 0.2, "high": 0.8},
        confidence=0.8,
        legend=["low", "high"],
    )
    assert is_correct(score, result, 1)
    assert not is_correct(score, result, 0)


def test_flatten_aligns_decisions_and_correctness():
    specs = {"u": NoulSpec("urgent?")}
    decisions = {
        "u": [NoulResult(noul=0.9, confidence=0.8), NoulResult(noul=0.1, confidence=0.8)]
    }
    examples = [{"labels": {"u": True}}, {"labels": {"u": True}}]
    flat, correct = flatten(specs, decisions, examples)
    assert len(flat) == 2
    assert correct.tolist() == [1.0, 0.0]


def test_selective_accuracy_at_fixed_coverages():
    scores = np.array([0.9, 0.8, 0.3, 0.2])
    correct = np.array([1.0, 1.0, 0.0, 0.0])
    quality = selective_accuracy(scores, correct, coverages=(0.5, 0.75, 1.0))
    assert quality["0.5"] == pytest.approx(1.0)  # принимаем оба правильных
    assert quality["0.8"] == pytest.approx(2 / 3)  # принято три из четырёх
    assert quality["1.0"] == pytest.approx(0.5)  # все решения


def _gate_row(**overrides):
    row = {
        "aurc": 0.30,
        "eaurc": 0.20,
        "coverage_at_5pct_risk": 0.50,
        "coverage_at_10pct_risk": 0.70,
        "acc_at_0.7_coverage": 0.85,
    }
    row.update(overrides)
    return row


def test_gate_accepts_when_candidate_not_worse():
    base = _gate_row()
    accepted, worse = gate_risk_vs_confidence(base, _gate_row(aurc=0.25))
    assert accepted
    assert worse == []


def test_gate_rejects_regression_on_single_metric():
    base = _gate_row()
    accepted, worse = gate_risk_vs_confidence(
        base, _gate_row(aurc=0.20, coverage_at_5pct_risk=0.40)
    )
    assert not accepted
    assert worse and worse[0].startswith("coverage_at_5pct_risk")


def test_gate_tolerates_noise_within_tol():
    base = _gate_row()
    accepted, _ = gate_risk_vs_confidence(base, _gate_row(aurc=0.3004), tol=0.0005)
    assert accepted
    # регрессия ровно на допуск + эпсилон — уже отказ
    accepted, _ = gate_risk_vs_confidence(base, _gate_row(aurc=0.3006), tol=0.0005)
    assert not accepted


def test_gate_respects_metric_direction():
    base = _gate_row()
    # AURC стал лучше, но точность при покрытии упала — всё равно отказ
    accepted, worse = gate_risk_vs_confidence(
        base, _gate_row(aurc=0.10, **{"acc_at_0.7_coverage": 0.80})
    )
    assert not accepted
    assert worse[0].startswith("acc_at_0.7_coverage")


def test_gate_loo_metrics_keys_match_run_loo_rows():
    keys = {metric.key for metric in GATE_LOO_METRICS}
    assert keys == {
        "loo_aurc",
        "loo_eaurc",
        "loo_coverage_at_5pct_risk",
        "loo_acc_at_0.7",
    }


def test_gate_calibration_only_where_it_is_resolvable():
    """Калибровка входит в гейт пула (in-dist/сертификат), но не одного вопроса.

    На одном новом вопросе измеренный разброс ECE-разницы сравним с эффектом
    (σ 0.03–0.07 при ~3 прогонах), поэтому постоянный допуск там отвергал бы на
    шуме; калибровку вернёт Шаг 3 по CI.
    """
    ab_keys = {metric.key for metric in GATE_AB_METRICS}
    assert {"ece", "brier"} <= ab_keys
    assert {"aurc", "eaurc", "coverage_at_5pct_risk", "acc_at_0.7_coverage"} <= ab_keys
    assert {"ece", "brier"} & {metric.key for metric in GATE_LOO_METRICS} == set()
    assert {"ece", "brier"} & {metric.key for metric in GATE_SHOTS_METRICS} == set()
    # у кривой тот же набор SLO-метрик, что у пула, только без калибровки
    assert {metric.key for metric in GATE_SHOTS_METRICS} == ab_keys - {"ece", "brier"}


def test_gate_calibration_metrics_have_measured_tolerances():
    """У калибровки свой допуск: бинаризованный ECE шумит сильнее AURC."""
    tolerances = {metric.key: metric.tol for metric in GATE_AB_METRICS}
    assert tolerances["aurc"] == pytest.approx(0.0005)
    assert tolerances["ece"] == pytest.approx(0.02)
    assert tolerances["brier"] == pytest.approx(0.01)

    # разница уровня seed-шума не должна отвергать вариант
    base = _gate_row(ece=0.20, brier=0.26)
    accepted, _ = gate_risk_vs_confidence(
        base, _gate_row(ece=0.212, brier=0.267)
    )
    assert accepted

    # а измеримая деградация калибровки — должна
    accepted, worse = gate_risk_vs_confidence(
        base, _gate_row(ece=0.25, brier=0.29)
    )
    assert not accepted
    assert any(reason.startswith("ece") for reason in worse)
    assert any(reason.startswith("brier") for reason in worse)


def test_gate_explicit_tol_overrides_all_metrics():
    base = _gate_row(ece=0.20, brier=0.26)
    accepted, _ = gate_risk_vs_confidence(
        base, _gate_row(ece=0.212, brier=0.267), tol=0.05
    )
    assert accepted
    accepted, worse = gate_risk_vs_confidence(
        base, _gate_row(ece=0.2004), tol=0.0001
    )
    assert not accepted and worse[0].startswith("ece")


def test_run_cli_reports_gate_in_json(tmp_path, capsys):
    """CLI показывает вердикт гейта и сохраняет его в metrics.json."""
    from eval.risk.run import main as run_main

    run_main([
        "--specs", "support", "--encoder", "hash", "--dim", "16",
        "--limit", "200", "--train-steps", "30", "--seed", "0",
        "--out-dir", str(tmp_path),
    ])
    out = capsys.readouterr().out
    assert "gate (risk vs confidence):" in out
    assert ("ПРИНЯТ" in out) or ("ОТКАЗАН" in out)

    payload = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    gate = payload["gate"]
    assert isinstance(gate["accepted"], bool)
    assert gate["tol"] == pytest.approx(0.0005)
    assert gate["tolerances"]["ece"] == pytest.approx(0.02)
    assert gate["tolerances"]["brier"] == pytest.approx(0.01)
    assert gate["baseline"] == "confidence"
    # калибровка P(ошибка) есть и у baseline, и у кандидата
    assert {"ece", "brier"} <= set(gate["metrics"]["confidence"])
    assert {"ece", "brier"} <= set(gate["metrics"]["risk_model"])
    if not gate["accepted"]:
        assert gate["worse"]

    # сертификат уезжает в risk.json — прод читает его при загрузке модели
    risk_payload = json.loads((tmp_path / "risk.json").read_text(encoding="utf-8"))
    assert risk_payload["gate"]["accepted"] == gate["accepted"]
    assert risk_payload["gate"]["worse"] == gate["worse"]
    from src.risk import ErrorRiskModel

    model = ErrorRiskModel.load(tmp_path / "risk.json")
    assert model.gate_rejected == (not gate["accepted"])


def test_bootstrap_mean_ci_brackets_mean_and_shrinks_with_n():
    values = [-0.05, -0.04, -0.06, -0.03, -0.05]
    ci = bootstrap_mean_ci(values, n_boot=400, seed=0)
    assert ci["n"] == 5
    assert ci["mean"] == pytest.approx(np.mean(values))
    assert ci["lo"] <= ci["mean"] <= ci["hi"]

    # у выборки без разброса интервал нулевой, у шумной — шире
    flat = bootstrap_mean_ci([0.1, 0.1, 0.1, 0.1], n_boot=200, seed=0)
    assert flat["lo"] == pytest.approx(0.1) and flat["hi"] == pytest.approx(0.1)
    noisy = bootstrap_mean_ci([0.0, 0.5, -0.5, 0.4, -0.4], n_boot=400, seed=0)
    assert noisy["hi"] - noisy["lo"] > flat["hi"] - flat["lo"]


def test_bootstrap_mean_ci_handles_empty():
    ci = bootstrap_mean_ci([])
    assert ci["n"] == 0 and np.isnan(ci["mean"])


def test_bootstrap_ranker_ci_covers_point_estimates():
    rng = np.random.default_rng(0)
    scores = rng.random(200)
    correct = (scores + rng.normal(0, 0.2, 200) > 0.5).astype(float)
    ci = bootstrap_ranker_ci(scores, correct, n_boot=120, seed=0)
    assert set(ci) == {"aurc", "eaurc", "coverage_at_5pct_risk",
                       "coverage_at_10pct_risk", "ece", "brier"}
    point = ranker_report(scores, correct)
    for key, entry in ci.items():
        assert entry["value"] == pytest.approx(point[key])
        assert entry["lo"] <= entry["hi"]
        assert entry["lo"] > -0.5 and entry["hi"] < 1.5


def test_paired_differences_flip_sign_for_higher_better():
    rows = [
        {"variant": "confidence", "seed": 0, "question": "q",
         "aurc": 0.30, "coverage_at_5pct_risk": 0.50},
        {"variant": "production", "seed": 0, "question": "q",
         "aurc": 0.40, "coverage_at_5pct_risk": 0.40},
    ]
    metrics = (
        GateMetric("aurc", True),
        GateMetric("coverage_at_5pct_risk", False),
    )
    diffs = paired_differences(rows, metrics)
    # «хуже» всегда положительно: у AURC production выше — хуже (+0.1),
    # у покрытия production ниже — тоже хуже (+0.1)
    assert diffs["aurc"] == pytest.approx([0.1])
    assert diffs["coverage_at_5pct_risk"] == pytest.approx([0.1])

    # если кандидата нет, пара пропускается
    only_baseline = [row for row in rows if row["variant"] == "confidence"]
    assert paired_differences(only_baseline, metrics)["aurc"] == []


def test_stability_report_flags_significant_regression():
    rows = [
        {"variant": "confidence", "seed": seed, "question": "q", "aurc": 0.30}
        for seed in range(6)
    ] + [
        {"variant": "production", "seed": seed, "question": "q", "aurc": 0.40}
        for seed in range(6)
    ]
    report = stability_report(paired_differences(rows, GATE_AB_METRICS))
    assert report["aurc"]["mean"] == pytest.approx(0.1)
    # разница одинакова во всех парах — интервал не накрывает ноль
    assert report["aurc"]["significant"] is True
    assert report["aurc"]["share_worse"] == pytest.approx(1.0)
    # калибровочных метрик в строках нет — их в отчёте тоже нет
    assert "ece" not in report
