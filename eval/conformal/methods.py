"""Альтернативные валидные методы выбора conformal-порога.

Продукт использует fixed-sequence (`src.conformal`): он даёт лучшую мощность
из проверенных схем. Здесь — два других способа, которые тоже контролируют
вероятность ложной сертификации, но платят за это иначе:

- ``bonferroni_grid`` — Бонферрони по фиксированной сетке: каждый кандидат
  проверяется на уровне ``delta / (grid_points * Q)``. Просто и устойчиво, но
  при малом ``delta`` граница настолько широкая, что порог часто не находится.
- ``split_half`` — порог выбирается на одной половине калибровки (любым
  правилом, хоть максимумом по покрытию), а граница считается на другой.
  Post-selection inference исчезает, потому что выбор не видит сертификационную
  половину; цена — вдвое меньше данных на границу.

Методы возвращают сертификат в той же форме, что ``selective_certificate``,
поэтому их можно прогнать через ``eval.conformal.core.evaluate_certificate``.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from src.conformal import clopper_pearson_upper


def _checked(
    scores: Any, correct: Any, groups: Any | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scores = np.asarray(scores, dtype=float).reshape(-1)
    correct = np.asarray(correct, dtype=float).reshape(-1)
    if scores.shape != correct.shape:
        raise ValueError(f"scores {scores.shape} != correct {correct.shape}")
    if scores.size == 0:
        raise ValueError("need at least one example")
    if not np.all((scores >= 0.0) & (scores <= 1.0)):
        raise ValueError("scores are P(error) and must be in [0, 1]")
    if not np.all((correct == 0.0) | (correct == 1.0)):
        raise ValueError("correct must be 0/1")
    if groups is None:
        labels = np.full(scores.size, "all", dtype=object)
    else:
        labels = np.asarray(groups, dtype=object).reshape(-1)
        if labels.size != scores.size:
            raise ValueError(f"groups {labels.shape} != scores {scores.shape}")
    return scores, correct, labels


def _group_bounds(
    scores: np.ndarray,
    correct: np.ndarray,
    groups: np.ndarray,
    threshold: float,
    delta: float,
) -> dict[str, dict[str, Any]]:
    accepted = scores <= threshold
    stats: dict[str, dict[str, Any]] = {}
    for name in sorted({str(label) for label in groups}):
        mask = groups == name
        taken = accepted & mask
        total = int(mask.sum())
        n = int(taken.sum())
        if n == 0:
            stats[name] = {
                "n": total,
                "accepted": 0,
                "coverage": 0.0,
                "risk": None,
                "upper_bound": 1.0,
            }
            continue
        errors = int((1.0 - correct[taken]).sum())
        stats[name] = {
            "n": total,
            "accepted": n,
            "coverage": n / total,
            "risk": errors / n,
            "upper_bound": clopper_pearson_upper(errors, n, delta),
        }
    return stats


def _certificate(
    method: str,
    threshold: float,
    scores: np.ndarray,
    correct: np.ndarray,
    groups: np.ndarray,
    *,
    alpha: float,
    delta: float,
    min_coverage: float,
    bound_delta: float | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    q = len(set(groups))
    level = delta / q if bound_delta is None else bound_delta / q
    if threshold < 0:
        return {
            "method": method,
            "threshold": -1.0,
            "coverage": 0.0,
            "empirical_risk": None,
            "upper_bound": 1.0,
            "alpha": float(alpha),
            "delta": float(delta),
            "min_coverage": float(min_coverage),
            **(extra or {}),
        }
    accepted = scores <= threshold
    stats = _group_bounds(scores, correct, groups, threshold, level)
    return {
        "method": method,
        "threshold": float(threshold),
        "coverage": float(accepted.mean()),
        "empirical_risk": float(1.0 - correct[accepted].mean()),
        "upper_bound": max(row["upper_bound"] for row in stats.values()),
        "alpha": float(alpha),
        "delta": float(delta),
        "min_coverage": float(min_coverage),
        "groups": stats,
        **(extra or {}),
    }


def bonferroni_grid(
    scores: Any,
    correct: Any,
    groups: Any | None = None,
    *,
    alpha: float,
    delta: float,
    min_coverage: float = 0.0,
    grid_points: int = 20,
) -> dict[str, Any]:
    """Порог с поправкой Бонферрони на ``grid_points`` кандидатов и вопросы."""
    if grid_points < 2:
        raise ValueError(f"grid_points must be >= 2, got {grid_points}")
    scores, correct, labels = _checked(scores, correct, groups)
    q = len(set(labels))
    per_test_delta = delta / (grid_points * q)
    best = None
    for threshold in np.linspace(0.0, 1.0, grid_points):
        accepted = scores <= threshold
        if accepted.mean() < min_coverage:
            continue
        stats = _group_bounds(scores, correct, labels, threshold, per_test_delta)
        if all(row["upper_bound"] <= alpha for row in stats.values()):
            best = _certificate(
                "bonferroni_grid",
                float(threshold),
                scores,
                correct,
                labels,
                alpha=alpha,
                delta=delta,
                min_coverage=min_coverage,
                bound_delta=per_test_delta,
                extra={
                    "grid_points": int(grid_points),
                    "per_test_delta": float(per_test_delta),
                },
            )
    return best or _certificate(
        "bonferroni_grid",
        -1.0,
        scores,
        correct,
        labels,
        alpha=alpha,
        delta=delta,
        min_coverage=min_coverage,
        extra={
            "grid_points": int(grid_points),
            "per_test_delta": float(per_test_delta),
        },
    )


def split_half(
    scores: Any,
    correct: Any,
    groups: Any | None = None,
    *,
    alpha: float,
    delta: float,
    min_coverage: float = 0.0,
    seed: int = 0,
    repeats: int = 20,
) -> dict[str, Any]:
    """Выбор порога на одной половине, сертификация на другой.

    ``repeats`` разбиений нужны только для диагностики: одно разбиение — один
    сертификат. Возвращает долю успешных разбиений и среднее покрытие среди
    них; ``coverage_mean`` считает провалы как 0.
    """
    if repeats < 1:
        raise ValueError(f"repeats must be >= 1, got {repeats}")
    scores, correct, labels = _checked(scores, correct, groups)
    q = len(set(labels))
    rng = np.random.default_rng(seed)
    runs: list[dict[str, Any]] = []
    for _ in range(repeats):
        order = rng.permutation(scores.size)
        half = scores.size // 2
        select, certify = order[:half], order[half:]
        select_scores, select_correct = scores[select], correct[select]
        rank = np.argsort(select_scores, kind="stable")
        sorted_correct = select_correct[rank]
        cumulative_errors = np.cumsum(1.0 - sorted_correct)
        accepted = np.arange(1, select_scores.size + 1)
        feasible = (cumulative_errors / accepted <= alpha) & (
            accepted / select_scores.size >= min_coverage
        )
        if not feasible.any():
            runs.append({"threshold": -1.0, "coverage": 0.0, "passed": False})
            continue
        index = int(np.max(np.nonzero(feasible)[0]))
        threshold = float(select_scores[rank][index])
        stats = _group_bounds(
            scores[certify], correct[certify], labels[certify], threshold, delta / q
        )
        taken = scores[certify] <= threshold
        passed = bool(taken.any()) and all(
            row["upper_bound"] <= alpha for row in stats.values()
        )
        runs.append(
            {
                "threshold": threshold,
                "coverage": float(taken.mean()) if taken.any() else 0.0,
                "passed": passed,
            }
        )
    coverages = [run["coverage"] for run in runs if run["passed"]]
    return {
        "method": "split_half",
        "repeats": int(repeats),
        "seed": int(seed),
        "pass_rate": sum(run["passed"] for run in runs) / len(runs),
        "coverage_mean": float(np.mean([run["coverage"] for run in runs])),
        "coverage_among_pass": float(np.mean(coverages)) if coverages else 0.0,
        "runs": runs,
    }
