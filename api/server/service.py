"""Сервис принятия решений: state + вопросы -> типизированные ответы."""
from __future__ import annotations

import hashlib
import logging

import torch
from torch import Tensor

from src.decision_head.heads import DecisionHeads
from src.encoder.base import StateEncoder
from src.sampler.parallel import sample_decisions
from src.sampler.questions import QuestionSpec
from src.schemas import DecisionResult

logger = logging.getLogger("decision.service")

FEATURE_DIM = 64


class HashStateEncoder:
    """Детерминированный fallback-энкодер: bag-of-токенов по хэшам."""

    def __init__(self, dim: int = FEATURE_DIM):
        self.dim = dim

    def encode(self, states: list[str]) -> Tensor:
        features = torch.zeros(len(states), self.dim)
        for row, text in enumerate(states):
            for token in text.lower().split():
                digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).hexdigest()
                features[row, int(digest, 16) % self.dim] += 1.0
            norm = features[row].norm()
            if norm > 0:
                features[row] /= norm
        return features


class DecisionService:
    def __init__(
        self,
        encoder: StateEncoder | None = None,
        dim: int = FEATURE_DIM,
        seed: int = 0,
    ):
        self.encoder = encoder or HashStateEncoder(dim)
        self.dim = dim
        self.seed = seed

    def decide(
        self, state: str, questions: dict[str, QuestionSpec]
    ) -> dict[str, DecisionResult]:
        features = self.encoder.encode([state])
        if features.shape[0] != len(questions):
            features = features.repeat(len(questions), 1)
        heads = DecisionHeads(features.shape[-1], questions, seed=self.seed)
        results = sample_decisions(heads, features)
        for name, result in results.items():
            dist = getattr(result, "probabilities", None) or {"p": result.noul}
            logger.info(
                "decision name=%s type=%s distribution=%s confidence=%.3f",
                name,
                result.type,
                dist,
                result.confidence,
            )
        return results