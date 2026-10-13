"""Тесты метрик калибровки — написаны до реализации (TDD RED)."""
import numpy as np
import pytest

from eval.calibration.metrics import (
    brier_score,
    calibration_curve,
    ece,
    multiclass_brier_score,
    reliability_curve,
    top_label_ece,
)


def test_perfect_calibration_zero_ece():
    probs = np.array([])
    outc = np.array([])
    for p in (0.1, 0.2, 0.3, 0.7, 0.8, 0.9):
        n = 20
        k = round(n * p)
        probs = np.concatenate([probs, np.full(n, p)])
        outc = np.concatenate([outc, np.array([1.0] * k + [0.0] * (n - k))])
    assert ece(probs, outc, n_bins=10) < 0.05


def test_perfect_calibration_zero_brier():
    probs = np.array([1.0, 0.0, 1.0, 0.0])
    outc = np.array([1.0, 0.0, 1.0, 0.0])
    assert brier_score(probs, outc) == pytest.approx(0.0, abs=1e-9)


def test_random_forecast_brier_quarter():
    probs = np.full(100, 0.5)
    outc = np.array([0.0, 1.0] * 50)
    assert brier_score(probs, outc) == pytest.approx(0.25, abs=1e-9)


def test_overconfident_has_positive_ece():
    probs = np.array([0.95, 0.95, 0.95, 0.95, 0.05, 0.05, 0.05, 0.05])
    outc = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
    assert ece(probs, outc, n_bins=10) > 0.1


def test_brier_lower_is_better():
    worse = brier_score(
        np.array([0.9, 0.9, 0.1, 0.1]), np.array([0.0, 0.0, 1.0, 1.0])
    )
    better = brier_score(
        np.array([0.6, 0.4, 0.6, 0.4]), np.array([0.0, 1.0, 0.0, 1.0])
    )
    assert better < worse


def test_ece_uses_confidence_binning_and_weights():
    probs = np.array([0.1, 0.2, 0.9, 0.8])
    outc = np.array([0.0, 0.0, 1.0, 1.0])
    bins = ece(probs, outc, n_bins=10, mode="confidence")
    assert 0.0 <= bins <= 1.0


def test_reliability_curve_shape():
    probs = np.array([0.1, 0.9, 0.2, 0.8])
    outc = np.array([0.0, 1.0, 0.0, 1.0])
    conf, acc, counts = reliability_curve(probs, outc, n_bins=10)
    assert len(conf) == len(acc) == len(counts) == 10
    assert (counts >= 0).all()
    assert (acc >= 0).all() and (acc <= 1.0).all()


def test_calibration_curve_returns_plot():
    probs = np.linspace(0.0, 1.0, 200)
    outc = (probs > 0.5).astype(float)
    fig = calibration_curve(probs, outc, n_bins=10)
    assert fig is not None
    import matplotlib.pyplot as plt

    plt.close(fig)


def test_multiclass_brier_perfect_is_zero():
    probs = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    labels = np.array([0, 1])
    assert multiclass_brier_score(probs, labels) == pytest.approx(0.0, abs=1e-9)


def test_multiclass_brier_uniform_two_classes():
    probs = np.full((4, 2), 0.5)
    labels = np.array([0, 1, 0, 1])
    # на каждый пример: 0.5^2 + 0.5^2 = 0.5
    assert multiclass_brier_score(probs, labels) == pytest.approx(0.5, abs=1e-9)


def test_multiclass_brier_lower_is_better():
    labels = np.array([0, 1])
    worse = multiclass_brier_score(np.array([[0.5, 0.5], [0.5, 0.5]]), labels)
    better = multiclass_brier_score(np.array([[0.9, 0.1], [0.1, 0.9]]), labels)
    assert better < worse


def test_top_label_ece_perfect_confidence_is_zero():
    probs = np.array([[1.0, 0.0], [0.0, 1.0]])
    labels = np.array([0, 1])
    assert top_label_ece(probs, labels) == pytest.approx(0.0, abs=1e-9)


def test_top_label_ece_overconfident_wrong():
    probs = np.array([[0.95, 0.05], [0.95, 0.05]])
    labels = np.array([1, 1])
    assert top_label_ece(probs, labels) > 0.5


def test_multiclass_metrics_reject_bad_shapes():
    with pytest.raises(ValueError):
        multiclass_brier_score(np.array([0.5, 0.5]), np.array([0]))
    with pytest.raises(ValueError):
        top_label_ece(np.array([0.5, 0.5]), np.array([0]))