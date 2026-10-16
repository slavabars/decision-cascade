"""Обучение оценщика риска ``P(ошибка)`` для risk-aware каскада.

Логистическая регрессия на признаках решения (состав — ``FEATURE_NAMES``,
см. ``src.risk.decision_features``). Обучается на **отложенном** наборе: решения
локальных голов и их фактическая корректность. Результат — ``ErrorRiskModel``,
который при обслуживании считается на numpy (тяжёлые зависимости не нужны).

Пример:
    python -m training.recipe.risk --help
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.risk import FEATURE_NAMES, ErrorRiskModel, features_matrix
from src.schemas import DecisionResult


def fit_error_risk_model(
    decisions: list[DecisionResult],
    correct: list[bool] | np.ndarray,
    *,
    seed: int = 0,
    C: float = 1.0,
) -> ErrorRiskModel:
    """Обучает ``P(ошибка)`` по решениям и их корректности.

    Если в выборке только один класс (все верны/все ошибочны), возвращает
    константную модель с соответствующей базовой вероятностью.
    """
    from sklearn.linear_model import LogisticRegression

    if len(decisions) != len(correct):
        raise ValueError("decisions and correct must have the same length")
    if len(decisions) < 2:
        raise ValueError("need at least 2 examples to fit a risk model")

    features = features_matrix(decisions)
    errors = 1 - np.asarray(correct, dtype=int).reshape(-1)
    if len(np.unique(errors)) < 2:
        eps = 1e-6
        p = float(np.clip(errors.mean(), eps, 1.0 - eps))
        return ErrorRiskModel(
            weights=np.zeros(len(FEATURE_NAMES)), bias=float(np.log(p / (1.0 - p)))
        )

    clf = LogisticRegression(C=C, random_state=seed, max_iter=1000)
    clf.fit(features, errors)
    return ErrorRiskModel(
        weights=clf.coef_[0],
        bias=float(clf.intercept_[0]),
        feature_names=FEATURE_NAMES,
    )


def fit_from_arrays(
    features: np.ndarray,
    correct: np.ndarray,
    *,
    seed: int = 0,
    C: float = 1.0,
) -> ErrorRiskModel:
    """Как ``fit_error_risk_model``, но по уже готовой матрице признаков.

    ``features`` должен идти в порядке ``FEATURE_NAMES``.
    """
    from sklearn.linear_model import LogisticRegression

    features = np.asarray(features, dtype=float)
    errors = 1 - np.asarray(correct, dtype=int).reshape(-1)
    if features.shape[0] != errors.shape[0]:
        raise ValueError("features rows != correct length")
    if features.shape[1] != len(FEATURE_NAMES):
        raise ValueError(
            f"features cols {features.shape[1]} != {len(FEATURE_NAMES)} "
            f"FEATURE_NAMES"
        )
    if len(np.unique(errors)) < 2:
        eps = 1e-6
        p = float(np.clip(errors.mean(), eps, 1.0 - eps))
        return ErrorRiskModel(
            weights=np.zeros(len(FEATURE_NAMES)),
            bias=float(np.log(p / (1.0 - p))),
        )
    clf = LogisticRegression(C=C, random_state=seed, max_iter=1000)
    clf.fit(features, errors)
    return ErrorRiskModel(
        weights=clf.coef_[0],
        bias=float(clf.intercept_[0]),
        feature_names=FEATURE_NAMES,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Fit an error-risk model")
    parser.add_argument(
        "--features", type=Path, required=True,
        help=f"npy [n, {len(FEATURE_NAMES)}] в порядке FEATURE_NAMES",
    )
    parser.add_argument("--correct", type=Path, required=True, help="npy [n] {0,1}")
    parser.add_argument("--out", type=Path, required=True, help="куда сохранить risk.json")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--C", type=float, default=1.0)
    args = parser.parse_args(argv)

    features = np.load(args.features)
    correct = np.load(args.correct)
    model = fit_from_arrays(features, correct, seed=args.seed, C=args.C)
    path = model.save(args.out)
    print(f"risk model -> {path}")


if __name__ == "__main__":
    main()
