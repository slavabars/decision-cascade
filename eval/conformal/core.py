"""Сертификат selective prediction: порог и гейт «не навреди».

Сертификат строится на калибровочной выборке (``src.conformal``), а здесь
проверяется на **отложенном тесте**: держится ли обещанная доля ошибок и не
хуже ли conformal-порог простого ранжирования по уверенности при том же
покрытии. Вердикт гейта уезжает в ``conformal.json``; прод при отказе
фолбэчит на порог уверенности, как risk-каскад при отклонённом сертификате.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from src.conformal import selective_certificate


def risk_at_coverage(
    scores: Any, correct: Any, coverage: float
) -> float:
    """Риск лучшего ранжирования по ``scores`` при заданном покрытии.

    ``scores`` — балл «решение верно» (больше — надёжнее). Аналог
    ``eval.calibration.selective.risk_at_coverage``, продублирован, чтобы
    ``eval.conformal`` не тянул лишние зависимости.
    """
    scores = np.asarray(scores, dtype=float).reshape(-1)
    correct = np.asarray(correct, dtype=float).reshape(-1)
    if scores.shape != correct.shape:
        raise ValueError(f"scores {scores.shape} != correct {correct.shape}")
    if scores.size == 0:
        raise ValueError("need at least one example")
    if not 0.0 <= coverage <= 1.0:
        raise ValueError(f"coverage must be in [0, 1], got {coverage}")
    order = np.argsort(-scores, kind="stable")
    errors = 1.0 - correct[order]
    risks = np.cumsum(errors) / np.arange(1, scores.size + 1)
    coverages = np.arange(1, scores.size + 1) / scores.size
    return float(
        np.interp(
            coverage,
            np.concatenate([[0.0], coverages]),
            np.concatenate([[0.0], risks]),
        )
    )


def evaluate_certificate(
    certificate: dict[str, Any],
    scores: Any,
    correct: Any,
    confidence_scores: Any,
    *,
    groups: Any | None = None,
    tol: float = 0.01,
) -> dict[str, Any]:
    """Гейт «не навреди»: conformal-порог против уверенности на тесте.

    Проверяет три вещи: покрытие не ниже заявленного минимума, эмпирический
    риск не выше ``alpha`` (с допуском ``tol``) и не хуже ранжирования по
    уверенности при том же покрытии. Отказ не отменяет гарантию калибровки, но
    означает, что на этом домене порог не стоит включать.

    ``groups`` — метки вопросов: по-вопросные числа теста кладутся в
    ``test["groups"]`` для диагностики, но вердикт гейта держится на общем
    риске: на коротком вопросе шум несравним с обещанием ``alpha``.
    """
    scores = np.asarray(scores, dtype=float).reshape(-1)
    correct = np.asarray(correct, dtype=float).reshape(-1)
    confidence_scores = np.asarray(confidence_scores, dtype=float).reshape(-1)
    if scores.shape != correct.shape or scores.shape != confidence_scores.shape:
        raise ValueError("scores, correct and confidence_scores must align")
    n = scores.size
    if n == 0:
        raise ValueError("need at least one example")
    threshold = float(certificate["threshold"])
    accepted = scores <= threshold
    coverage = float(accepted.mean())
    if accepted.any():
        risk = float(1.0 - correct[accepted].mean())
    else:
        risk = 1.0
    baseline_risk = risk_at_coverage(confidence_scores, correct, coverage)
    alpha = float(certificate["alpha"])
    min_coverage = float(certificate.get("min_coverage", 0.0))

    worse: list[str] = []
    if coverage <= 0.0:
        worse.append("coverage is zero: сертификат не принимает ни одного решения")
    elif coverage < min_coverage:
        worse.append(
            f"coverage {coverage:.3f} < min_coverage {min_coverage:.3f}"
        )
    if risk > alpha + tol:
        worse.append(f"selective risk {risk:.4f} > alpha {alpha:.4f} + tol {tol:g}")
    if risk > baseline_risk + tol:
        worse.append(
            f"selective risk {risk:.4f} > confidence {baseline_risk:.4f} "
            f"+ tol {tol:g}"
        )
    test: dict[str, Any] = {
        "n": int(n),
        "coverage": coverage,
        "risk": risk,
        "accuracy": 1.0 - risk,
        "baseline_risk": baseline_risk,
    }
    if groups is not None:
        labels = np.asarray(groups, dtype=object).reshape(-1)
        if labels.size != n:
            raise ValueError(f"groups {labels.shape} != scores {scores.shape}")
        group_rows: dict[str, dict[str, Any]] = {}
        for name in sorted({str(label) for label in labels}):
            mask = labels == name
            group_accepted = accepted & mask
            group_rows[name] = {
                "n": int(mask.sum()),
                "accepted": int(group_accepted.sum()),
                "coverage": float(group_accepted.sum() / mask.sum()),
                "risk": (
                    float(1.0 - correct[group_accepted].mean())
                    if group_accepted.any()
                    else None
                ),
            }
        test["groups"] = group_rows
    return {
        "accepted": not worse,
        "baseline": "confidence",
        "tol": tol,
        "worse": worse,
        "test": test,
    }


def fit_certificate(
    model: Any,
    decisions: list[Any],
    correct: Any,
    *,
    alpha: float,
    delta: float,
    min_coverage: float,
    groups: Any | None = None,
) -> dict[str, Any]:
    """Посчитать ``P(ошибка)`` на калибровке и вернуть сертификат."""
    scores = np.asarray(
        [model.predict_error(decision) for decision in decisions], dtype=float
    )
    return selective_certificate(
        scores,
        correct,
        groups=groups,
        alpha=alpha,
        delta=delta,
        min_coverage=min_coverage,
    )
