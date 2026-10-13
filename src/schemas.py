"""Типизированные примитивы решений и правила валидации вероятностей."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

SCHEMA_VERSION = "1"


def normalize_probabilities(weights: dict[str, float]) -> dict[str, float]:
    if not weights:
        raise ValueError("probability weights must not be empty")
    if any(w < 0.0 for w in weights.values()):
        raise ValueError("probability weights must be non-negative")
    total = sum(weights.values())
    if total <= 0.0:
        raise ValueError("probability weights must have a positive sum")
    return {k: v / total for k, v in weights.items()}


def validate_noul(noul: float) -> float:
    if not 0.0 <= noul <= 1.0:
        raise ValueError(f"noul must be in [0.0, 1.0], got {noul}")
    return noul


def validate_distribution(
    probabilities: dict[str, float], options: tuple[str, ...]
) -> dict[str, float]:
    if set(probabilities) != set(options):
        raise ValueError("probabilities keys must match the option set exactly")
    if any(p < 0.0 for p in probabilities.values()):
        raise ValueError("probabilities must be non-negative")
    total = sum(probabilities.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"probabilities must sum to 1.0, got {total}")
    return dict(probabilities)


def confidence_noul(noul: float) -> float:
    return float(abs(2.0 * noul - 1.0))


def confidence_max(probabilities: dict[str, float]) -> float:
    return float(max(probabilities.values()))


@dataclass(frozen=True)
class NoulResult:
    type: Literal["noul"] = "noul"
    noul: float = 0.0
    confidence: float = 0.0


@dataclass(frozen=True)
class ChoiceResult:
    type: Literal["choice"] = "choice"
    choice: str = ""
    probabilities: dict[str, float] = None
    confidence: float = 0.0


@dataclass(frozen=True)
class ScoreResult:
    type: Literal["score"] = "score"
    score: float = 0.0
    probabilities: dict[str, float] = None
    confidence: float = 0.0
    legend: tuple[str, ...] = ()


DecisionResult = NoulResult | ChoiceResult | ScoreResult