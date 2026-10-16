"""Оценка вероятности ошибки решения — обучение отказу (selective prediction).

Каскад может эскалировать не по «сырой» уверенности, а по оценке вероятности
ошибки локального решения. Модель — логистическая регрессия на признаках решения
(уверенность, нормированная энтропия, отрыв лучшего класса); обучается на
отложенном наборе (``training.recipe.risk``), а при обслуживании считается на
numpy — то есть в пути инференса нет тяжёлых зависимостей.
"""
from __future__ import annotations

import json
import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.schemas import DecisionResult

FEATURE_NAMES = ("confidence", "entropy", "margin")


def decision_distribution(decision: DecisionResult) -> np.ndarray:
    """Распределение вероятностей решения как вектор (noul -> [p, 1 - p])."""
    if decision.type == "noul":
        p = float(decision.noul)
        return np.asarray([p, 1.0 - p], dtype=float)
    probabilities = getattr(decision, "probabilities", None)
    if probabilities:
        values = np.asarray([float(v) for v in probabilities.values()], dtype=float)
    else:
        values = np.asarray([float(decision.confidence)], dtype=float)
    total = values.sum()
    return values / total if total > 0.0 else values


def decision_features(decision: DecisionResult) -> np.ndarray:
    """Признаки решения для оценщика риска: ``[confidence, entropy, margin]``."""
    dist = decision_distribution(decision)
    order = np.sort(dist)[::-1]
    confidence = float(order[0])
    if dist.size > 1:
        entropy = float(
            -(dist * np.log(np.clip(dist, 1e-12, 1.0))).sum() / math.log(dist.size)
        )
        margin = float(order[0] - order[1])
    else:
        entropy = 0.0
        margin = 1.0
    return np.asarray([confidence, entropy, margin], dtype=float)


def features_matrix(decisions: Iterable[DecisionResult]) -> np.ndarray:
    return np.vstack([decision_features(d) for d in decisions])


def _sigmoid(z: float) -> float:
    if z >= 0.0:
        return 1.0 / (1.0 + math.exp(-z))
    exp_z = math.exp(z)
    return exp_z / (1.0 + exp_z)


@dataclass
class ErrorRiskModel:
    """Логистический оценщик ``P(ошибка)`` по признакам решения."""

    weights: np.ndarray
    bias: float
    feature_names: tuple[str, ...] = FEATURE_NAMES

    def __post_init__(self) -> None:
        self.weights = np.asarray(self.weights, dtype=float).reshape(-1)
        if self.weights.size != len(self.feature_names):
            raise ValueError(
                f"weights size {self.weights.size} != "
                f"feature_names {len(self.feature_names)}"
            )

    def predict_error(self, decision: DecisionResult) -> float:
        """Вероятность того, что решение неверно (0..1)."""
        z = float(np.dot(self.weights, decision_features(decision)) + self.bias)
        return _sigmoid(z)

    def should_escalate(self, decision: DecisionResult, risk_budget: float) -> bool:
        return self.predict_error(decision) > risk_budget

    def to_dict(self) -> dict:
        return {
            "feature_names": list(self.feature_names),
            "weights": [float(w) for w in self.weights],
            "bias": float(self.bias),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> ErrorRiskModel:
        return cls(
            weights=np.asarray(payload["weights"], dtype=float),
            bias=float(payload["bias"]),
            feature_names=tuple(payload.get("feature_names", FEATURE_NAMES)),
        )

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> ErrorRiskModel:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
