"""Оценка вероятности ошибки решения — обучение отказу (selective prediction).

Каскад может эскалировать не по «сырой» уверенности, а по оценке вероятности
ошибки локального решения. Модель — логистическая регрессия на признаках
решения (уверенность, нормированная энтропия, отрыв лучшего класса, разрыв
логитов двух лучших исходов, число исходов и знаковый логит `noul`;
подробности — в ``decision_features``); обучается на отложенном наборе
(``training.recipe.risk``), а при обслуживании считается на numpy — то есть в
пути инференса нет тяжёлых зависимостей.
"""
from __future__ import annotations

import json
import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.schemas import DecisionResult

FEATURE_NAMES = (
    "confidence",
    "entropy",
    "margin",
    "log_odds_gap",
    "n_classes",
    "noul_logit",
)

#: Состав признаков моделей, обученных до расширения (файлы ``risk.json`` с
#: тремя весами). Такие модели продолжают работать в прежней семантике:
#: ``predict_error`` берёт по именам только те признаки, на которых модель
#: обучена.
LEGACY_FEATURE_NAMES = ("confidence", "entropy", "margin")


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
    """Признаки решения для оценщика риска (порядок — ``FEATURE_NAMES``).

    Первые три выведены из нормированного распределения; последние три добавлены
    по замеру ``eval/risk/features.py`` (AURC −3…−31% против трёх признаков,
    см. `docs/risk-aware-cascade.md`):

    - ``log_odds_gap`` — `ln(p_1 / p_2)` двух лучших исходов. Для `k >= 3` это
      **разрыв сырых логитов до softmax** (свойство softmax: `logit_i - logit_j =
      ln(p_i / p_j)`), то есть «сила улики», несводимая к нормированному
      распределению; для `k = 2` детерминированно вытекает из уверенности и
      вклада не даёт.
    - ``n_classes`` — число исходов решения (для `noul` — 2).
    - ``noul_logit`` — знаковый логит `noul`: первые три признака симметричны
      относительно `p <-> 1-p`, поэтому знак теряется.

    Для `noul` распределение — это `[p, 1-p]`.
    """
    dist = decision_distribution(decision)
    order = np.sort(dist)[::-1]
    confidence = float(order[0])
    if dist.size > 1:
        entropy = float(
            -(dist * np.log(np.clip(dist, 1e-12, 1.0))).sum() / math.log(dist.size)
        )
        margin = float(order[0] - order[1])
        log_odds_gap = float(
            np.log(np.clip(order[0], 1e-12, None))
            - np.log(np.clip(order[1], 1e-12, None))
        )
    else:
        entropy = 0.0
        margin = 1.0
        log_odds_gap = 0.0
    noul_logit = 0.0
    if decision.type == "noul":
        p = float(np.clip(decision.noul, 1e-12, 1.0 - 1e-12))
        noul_logit = float(np.log(p / (1.0 - p)))
    return np.asarray(
        [confidence, entropy, margin, log_odds_gap, float(dist.size), noul_logit],
        dtype=float,
    )


def features_matrix(decisions: Iterable[DecisionResult]) -> np.ndarray:
    return np.vstack([decision_features(d) for d in decisions])


def _sigmoid(z: float) -> float:
    if z >= 0.0:
        return 1.0 / (1.0 + math.exp(-z))
    exp_z = math.exp(z)
    return exp_z / (1.0 + exp_z)


@dataclass
class ErrorRiskModel:
    """Логистический оценщик ``P(ошибка)`` по признакам решения.

    ``feature_names`` — имена признаков в том порядке, в каком они идут в
    ``weights``; все должны входить в ``FEATURE_NAMES``. Модели старого формата
    (три признака, ``LEGACY_FEATURE_NAMES``) загружаются и работают как прежде.
    """

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
        unknown = [name for name in self.feature_names if name not in FEATURE_NAMES]
        if unknown:
            raise ValueError(
                f"unknown feature_names {unknown}; known: {list(FEATURE_NAMES)}"
            )

    def vector(self, decision: DecisionResult) -> np.ndarray:
        """Вектор признаков решения в порядке ``self.feature_names``."""
        features = dict(zip(FEATURE_NAMES, decision_features(decision)))
        return np.asarray([features[name] for name in self.feature_names],
                          dtype=float)

    def predict_error(self, decision: DecisionResult) -> float:
        """Вероятность того, что решение неверно (0..1)."""
        z = float(np.dot(self.weights, self.vector(decision)) + self.bias)
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
        weights = np.asarray(payload["weights"], dtype=float)
        names = payload.get("feature_names")
        if names is None:
            # файл без списка признаков: определяем состав по числу весов, иначе
            # трёхвесовый ``risk.json`` из старого формата не загрузится
            names = (
                LEGACY_FEATURE_NAMES
                if weights.size == len(LEGACY_FEATURE_NAMES)
                else FEATURE_NAMES
            )
        return cls(
            weights=weights,
            bias=float(payload["bias"]),
            feature_names=tuple(names),
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
