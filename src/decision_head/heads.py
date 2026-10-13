"""Головы решений: отображение признаков в типизированные вероятности."""
from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor, nn

from src.sampler.questions import (
    ChoiceSpec,
    NoulSpec,
    QuestionSpec,
    ScoreSpec,
    spec_from_dict,
    spec_to_dict,
)
from src.schemas import (
    ChoiceResult,
    DecisionResult,
    NoulResult,
    ScoreResult,
    confidence_max,
    confidence_noul,
    validate_distribution,
)


def _require_unique(name: str, field: str, values: tuple[str, ...]) -> None:
    """Опечатки в опциях/уровнях ломают распределение — ловим на входе."""
    if len(set(values)) != len(values):
        raise ValueError(
            f"question {name!r} {field} must be unique; got {list(values)}"
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
        self.dim = dim
        self.questions = dict(questions)
        self.heads: dict[str, nn.Module] = {}
        if seed is not None:
            torch.manual_seed(seed)
        for name, spec in self.questions.items():
            if isinstance(spec, NoulSpec):
                self.heads[name] = NoulHead(dim)
            elif isinstance(spec, ChoiceSpec):
                _require_unique(name, "options", spec.options)
                self.heads[name] = ChoiceHead(dim, spec.options)
            elif isinstance(spec, ScoreSpec):
                _require_unique(name, "levels", spec.levels)
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

    def save_checkpoint(self, path: str | Path) -> Path:
        """Сохраняет веса голов вместе со спецификациями вопросов для восстановления."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "dim": self.dim,
                "questions": {
                    name: spec_to_dict(spec) for name, spec in self.questions.items()
                },
                "state_dict": self.state_dict(),
            },
            path,
        )
        return path

    @classmethod
    def load_checkpoint(cls, path: str | Path, seed: int | None = None) -> DecisionHeads:
        """Восстанавливает головы (структуру и веса) из чекпоинта."""
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
        questions = {
            name: spec_from_dict(data) for name, data in payload["questions"].items()
        }
        heads = cls(payload["dim"], questions, seed=seed)
        heads.load_state_dict(payload["state_dict"])
        return heads