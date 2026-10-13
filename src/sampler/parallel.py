"""Параллельный семплер: все вопросы за один прогон по фичам состояния."""
from __future__ import annotations

from torch import Tensor

from src.decision_head.heads import DecisionHeads
from src.schemas import DecisionResult


def sample_decisions(
    heads: DecisionHeads, features: Tensor
) -> dict[str, DecisionResult]:
    results = heads.evaluate(features)
    return {name: result for name, result in results.items()}