"""Тесты честного сравнения маршрутизации (eval/risk/core)."""
import numpy as np
import pytest

from eval.risk.core import (
    cascade_summary,
    flatten,
    is_correct,
    labels_for,
    ranker_report,
    risk_escalation,
    selective_accuracy,
    threshold_escalation,
)
from src.risk import ErrorRiskModel
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
    model = ErrorRiskModel(weights=np.array([-20.0, 0.0, 0.0]), bias=14.0)
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
