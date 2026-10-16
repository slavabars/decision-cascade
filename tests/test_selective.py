"""Тесты selective prediction: риск–покрытие, AURC/eAURC, покрытие при бюджете риска."""
import numpy as np
import pytest

from eval.calibration.selective import (
    aurc,
    coverage_at_risk,
    eaurc,
    optimal_aurc,
    risk_at_coverage,
    risk_coverage_curve,
)


def test_perfect_ranker_curve_is_monotone_non_increasing_risk():
    # уверенность совпадает с корректностью -> риск не растёт
    scores = np.array([0.9, 0.8, 0.3, 0.2])
    correct = np.array([1, 1, 0, 0])
    coverages, risks = risk_coverage_curve(scores, correct)
    assert coverages[0] == 0.0 and risks[0] == 0.0
    assert np.all(np.diff(risks) >= -1e-12)
    assert risks[-1] == pytest.approx(0.5)


def test_perfect_ranker_aurc_is_minimal():
    scores = np.array([0.9, 0.8, 0.3, 0.2])
    correct = np.array([1, 1, 0, 0])
    # идеальное ранжирование совпадает с оракулом -> нулевой превышающий AURC
    assert aurc(scores, correct) == pytest.approx(optimal_aurc(correct))
    assert eaurc(scores, correct) == pytest.approx(0.0, abs=1e-12)


def test_worst_ranker_has_positive_excess_aurc():
    scores = np.array([0.9, 0.8, 0.3, 0.2])
    correct = np.array([0, 0, 1, 1])  # инвертировано
    assert aurc(scores, correct) > optimal_aurc(correct)
    assert eaurc(scores, correct) > 0.0


def test_risk_at_coverage_endpoints():
    scores = np.array([0.9, 0.8, 0.3, 0.2])
    correct = np.array([1, 1, 0, 0])
    assert risk_at_coverage(scores, correct, 0.0) == 0.0
    assert risk_at_coverage(scores, correct, 1.0) == pytest.approx(0.5)
    assert risk_at_coverage(scores, correct, 0.5) == 0.0


def test_coverage_at_risk_respects_budget():
    scores = np.array([0.9, 0.8, 0.3, 0.2])
    correct = np.array([1, 1, 0, 0])
    # при нулевом бюджете принимаем только безошибочные -> 2 из 4
    assert coverage_at_risk(scores, correct, 0.0) == pytest.approx(0.5)
    assert coverage_at_risk(scores, correct, 0.5) == pytest.approx(1.0)


def test_empty_input_rejected():
    with pytest.raises(ValueError):
        risk_coverage_curve(np.array([]), np.array([]))


def test_shape_mismatch_rejected():
    with pytest.raises(ValueError):
        risk_coverage_curve(np.array([0.1, 0.2]), np.array([1]))


def test_risk_budget_bounds_validated():
    scores = np.array([0.9, 0.2])
    correct = np.array([1, 0])
    with pytest.raises(ValueError):
        coverage_at_risk(scores, correct, 1.5)
    with pytest.raises(ValueError):
        risk_at_coverage(scores, correct, -0.1)
