"""Selective prediction: risk–coverage кривые и площадь под ними (AURC/eAURC).

Когда решение можно отложить (передать сильной модели), качество измеряют не
одним числом accuracy, а компромиссом «покрытие — риск»: при какой доле
принятых решений какова частота ошибок среди них. ``aurc`` (area under the
risk–coverage curve) суммирует этот компромисс одним числом, а ``eaurc`` —
превышение над оптимумом (идеальный ранжировщик). Чем ниже, тем лучше.

``scores`` — уверенность/оценка «быть правильным» (больше — надёжнее).
"""
from __future__ import annotations

import numpy as np


def risk_coverage_curve(
    scores: np.ndarray, correct: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Кривая риск–покрытие: принимаем сначала самые уверенные решения.

    Возвращает ``(coverages, risks)`` длины ``n + 1`` с точкой ``(0, 0)``:
    ``coverage`` пробегает ``0, 1/n, ..., 1``, ``risk`` — доля ошибок среди
    принятых при этом покрытии.
    """
    scores = np.asarray(scores, dtype=float).reshape(-1)
    correct = np.asarray(correct, dtype=float).reshape(-1)
    if scores.shape != correct.shape:
        raise ValueError(f"scores {scores.shape} != correct {correct.shape}")
    n = scores.size
    if n == 0:
        raise ValueError("need at least one example")
    order = np.argsort(-scores, kind="stable")
    errors = 1.0 - correct[order]
    risks = np.cumsum(errors) / np.arange(1, n + 1)
    coverages = np.arange(1, n + 1) / n
    return np.concatenate([[0.0], coverages]), np.concatenate([[0.0], risks])


def aurc(scores: np.ndarray, correct: np.ndarray) -> float:
    """Area under the risk–coverage curve (ниже — лучше)."""
    coverages, risks = risk_coverage_curve(scores, correct)
    return float(np.trapezoid(risks, coverages))


def optimal_risk_coverage(correct: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Оракул: сначала принимаем все правильные решения, затем все ошибочные."""
    correct = np.asarray(correct, dtype=float).reshape(-1)
    n = correct.size
    if n == 0:
        raise ValueError("need at least one example")
    ordered = np.sort(correct)[::-1]
    errors = 1.0 - ordered
    risks = np.cumsum(errors) / np.arange(1, n + 1)
    coverages = np.arange(1, n + 1) / n
    return np.concatenate([[0.0], coverages]), np.concatenate([[0.0], risks])


def optimal_aurc(correct: np.ndarray) -> float:
    coverages, risks = optimal_risk_coverage(correct)
    return float(np.trapezoid(risks, coverages))


def eaurc(scores: np.ndarray, correct: np.ndarray) -> float:
    """Excess AURC: ``aurc - optimal_aurc`` (0 — идеальный ранжировщик)."""
    return aurc(scores, correct) - optimal_aurc(correct)


def risk_at_coverage(
    scores: np.ndarray, correct: np.ndarray, coverage: float
) -> float:
    """Риск (частота ошибок среди принятых) при заданном покрытии."""
    if not 0.0 <= coverage <= 1.0:
        raise ValueError(f"coverage must be in [0, 1], got {coverage}")
    coverages, risks = risk_coverage_curve(scores, correct)
    return float(np.interp(coverage, coverages, risks))


def coverage_at_risk(
    scores: np.ndarray, correct: np.ndarray, max_risk: float
) -> float:
    """Наибольшее покрытие, при котором риск не превышает ``max_risk``.

    Это и есть «бюджет риска»: сколько решений можно принять локально, если
    допускаем долю ошибок не выше ``max_risk``.
    """
    if not 0.0 <= max_risk <= 1.0:
        raise ValueError(f"max_risk must be in [0, 1], got {max_risk}")
    coverages, risks = risk_coverage_curve(scores, correct)
    allowed = np.nonzero(risks <= max_risk)[0]
    if allowed.size == 0:
        return 0.0
    return float(coverages[allowed[-1]])
