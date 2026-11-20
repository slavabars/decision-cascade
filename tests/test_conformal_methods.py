"""Альтернативные методы conformal-порога: валидность и мощность."""
from __future__ import annotations

import numpy as np
import pytest

from eval.conformal.methods import bonferroni_grid, split_half


def _separable(n: int = 500) -> tuple[np.ndarray, np.ndarray]:
    scores = np.linspace(0.0, 1.0, n)
    # низкий P(ошибка) -> верно, высокий -> ошибка
    correct = (scores <= 0.5).astype(float)
    return scores, correct


def test_bonferroni_grid_finds_coverage_when_signal_exists():
    scores, correct = _separable()

    certificate = bonferroni_grid(
        scores, correct, alpha=0.1, delta=0.1, min_coverage=0.3, grid_points=20
    )

    assert certificate["method"] == "bonferroni_grid"
    assert certificate["coverage"] >= 0.3
    assert certificate["upper_bound"] <= 0.1


def test_bonferroni_grid_controls_false_certification():
    rng = np.random.default_rng(11)
    trials, n = 300, 500
    errors = 0
    for _ in range(trials):
        scores = rng.uniform(0.0, 1.0, n)
        correct = (rng.uniform(0.0, 1.0, n) > 0.0501).astype(float)
        certificate = bonferroni_grid(
            scores, correct, alpha=0.05, delta=0.1, min_coverage=0.3, grid_points=20
        )
        if certificate["coverage"] > 0:
            errors += 1
    assert errors / trials < 0.1


def test_split_half_finds_coverage_when_signal_exists():
    scores, correct = _separable()

    result = split_half(
        scores, correct, alpha=0.1, delta=0.1, min_coverage=0.3, seed=0, repeats=10
    )

    assert result["pass_rate"] > 0
    assert result["coverage_among_pass"] >= 0.3


def test_split_half_is_deterministic():
    scores, correct = _separable()
    first = split_half(scores, correct, alpha=0.1, delta=0.1, seed=7, repeats=5)
    second = split_half(scores, correct, alpha=0.1, delta=0.1, seed=7, repeats=5)

    assert first == second


def test_methods_validate_inputs():
    scores, correct = _separable()
    with pytest.raises(ValueError, match="grid_points"):
        bonferroni_grid(scores, correct, alpha=0.1, delta=0.1, grid_points=1)
    with pytest.raises(ValueError, match="repeats"):
        split_half(scores, correct, alpha=0.1, delta=0.1, repeats=0)
    with pytest.raises(ValueError, match="scores"):
        bonferroni_grid(scores, correct[:-1], alpha=0.1, delta=0.1)


def test_compare_reports_all_three_methods():
    from eval.conformal.compare import compare

    scores, correct = _separable()
    groups = np.asarray(["a", "b"] * (scores.size // 2))
    data = {
        "calibration_scores": scores[:250],
        "calibration_correct": correct[:250],
        "calibration_groups": groups[:250],
        "test_scores": scores[250:],
        "test_correct": correct[250:],
        "test_groups": groups[250:],
        "confidence_scores": 1.0 - scores[250:],
    }

    rows = compare(
        data,
        alphas=[0.1],
        delta=0.1,
        min_coverages=[0.3],
        grid_points=[20],
        split_seed=0,
        split_repeats=3,
    )

    assert {row["method"] for row in rows} == {
        "fixed_sequence",
        "bonferroni_grid_20",
        "split_half",
    }
    fixed = next(row for row in rows if row["method"] == "fixed_sequence")
    assert fixed["calibration_coverage"] > 0
    assert fixed["test_coverage"] > 0
