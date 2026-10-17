"""Тесты честного сравнения маршрутизации (eval/risk/core)."""
import json

import numpy as np
import pytest

from eval.risk.core import (
    GATE_LOO_METRICS,
    cascade_summary,
    flatten,
    gate_risk_vs_confidence,
    is_correct,
    labels_for,
    ranker_report,
    risk_escalation,
    selective_accuracy,
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
    }
    assert report["eaurc"] == pytest.approx(0.0, abs=1e-9)
    assert 0.0 <= report["coverage_at_5pct_risk"] <= 1.0


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
    keys = {key for key, _ in GATE_LOO_METRICS}
    assert keys == {
        "loo_aurc",
        "loo_eaurc",
        "loo_coverage_at_5pct_risk",
        "loo_acc_at_0.7",
    }


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
    assert gate["baseline"] == "confidence"
    if not gate["accepted"]:
        assert gate["worse"]

    # сертификат уезжает в risk.json — прод читает его при загрузке модели
    risk_payload = json.loads((tmp_path / "risk.json").read_text(encoding="utf-8"))
    assert risk_payload["gate"]["accepted"] == gate["accepted"]
    assert risk_payload["gate"]["worse"] == gate["worse"]
    from src.risk import ErrorRiskModel

    model = ErrorRiskModel.load(tmp_path / "risk.json")
    assert model.gate_rejected == (not gate["accepted"])
