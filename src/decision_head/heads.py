"""Головы решений: отображение признаков в типизированные вероятности."""
from __future__ import annotations

import torch
from torch import Tensor, nn

from src.sampler.questions import ChoiceSpec, NoulSpec, QuestionSpec, ScoreSpec
from src.schemas import (
    ChoiceResult,
    DecisionResult,
    NoulResult,
    ScoreResult,
    confidence_max,
    confidence_noul,
    validate_distribution,
)


class NoulHead(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.linear = nn.Linear(dim, 1)

    def logits(self, features: Tensor) -> Tensor:
        return self.linear(features).squeeze(-1)

    def forward(self, features: Tensor) -> NoulResult:
        logit = torch.sigmoid(self.logits(features))
        p = float(logit.item())
        return NoulResult(noul=p, confidence=confidence_noul(p))


class ChoiceHead(nn.Module):
    def __init__(self, dim: int, options: tuple[str, ...]):
        super().__init__()
        self.options = tuple(options)
        self.linear = nn.Linear(dim, len(self.options))

    def logits(self, features: Tensor) -> Tensor:
        return self.linear(features)

    def forward(self, features: Tensor) -> ChoiceResult:
        probs = torch.softmax(self.logits(features).squeeze(0), dim=-1)
        probs_dict = {opt: float(p) for opt, p in zip(self.options, probs.tolist())}
        validate_distribution(probs_dict, self.options)
        choice = max(self.options, key=lambda o: probs_dict[o])
        return ChoiceResult(
            choice=choice,
            probabilities=probs_dict,
            confidence=confidence_max(probs_dict),
        )


class ScoreHead(nn.Module):
    def __init__(self, dim: int, levels: tuple[str, ...]):
        super().__init__()
        self.levels = tuple(levels)
        self.linear = nn.Linear(dim, len(self.levels))

    def logits(self, features: Tensor) -> Tensor:
        return self.linear(features)

    def forward(self, features: Tensor) -> ScoreResult:
        probs_list = torch.softmax(self.logits(features).squeeze(0), dim=-1).tolist()
        probs_dict = {lvl: float(p) for lvl, p in zip(self.levels, probs_list)}
        validate_distribution(probs_dict, self.levels)
        score = float(sum(i * p for i, p in enumerate(probs_list)))
        return ScoreResult(
            score=score,
            probabilities=probs_dict,
            confidence=confidence_max(probs_dict),
            legend=self.levels,
        )


class DecisionHeads(nn.Module):
    def __init__(
        self, dim: int, questions: dict[str, QuestionSpec], seed: int | None = None
    ):
        super().__init__()
        self.questions = dict(questions)
        self.heads: dict[str, nn.Module] = {}
        if seed is not None:
            torch.manual_seed(seed)
        for name, spec in self.questions.items():
            if isinstance(spec, NoulSpec):
                self.heads[name] = NoulHead(dim)
            elif isinstance(spec, ChoiceSpec):
                self.heads[name] = ChoiceHead(dim, spec.options)
            elif isinstance(spec, ScoreSpec):
                self.heads[name] = ScoreHead(dim, spec.levels)
            else:
                raise TypeError(f"unsupported question spec: {type(spec).__name__}")
            self.add_module(name, self.heads[name])

    def logits(self, features: Tensor) -> dict[str, Tensor]:
        if features.dim() != 2:
            raise ValueError(f"features must be [batch, dim]; got {tuple(features.shape)}")
        return {name: head.logits(features) for name, head in self.heads.items()}

    def evaluate(self, features: Tensor) -> dict[str, DecisionResult]:
        if features.dim() != 2 or features.shape[0] != len(self.questions):
            raise ValueError(
                f"features must be [n_questions, dim]; got shape {tuple(features.shape)}"
            )
        results = {}
        for i, (name, head) in enumerate(self.heads.items()):
            results[name] = head(features[i].unsqueeze(0))
        return results