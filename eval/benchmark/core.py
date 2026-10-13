"""Офлайн-бенчмарк голов решений: обучение (опц.) и метрики на датасете.

Признаки строятся по-вопросно (``build_prompt(state, spec)``), как в сервисе;
обучение идёт через ``SFTTrainer`` с признаками по вопросам, поэтому каждая
голова учится на своих промптах без кросс-вопросного смешивания целей.
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np
import torch

from eval.calibration.metrics import brier_score, ece
from src.decision_head.heads import DecisionHeads
from src.sampler.questions import (
    ChoiceSpec,
    NoulSpec,
    QuestionSpec,
    ScoreSpec,
    build_prompt,
)
from training.sft.train import SFTTrainer


class TextEncoder(Protocol):
    def encode(self, states: list[str]) -> torch.Tensor: ...


def noul_metrics(probs: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    probs = np.asarray(probs, dtype=float).reshape(-1)
    labels = np.asarray(labels, dtype=float).reshape(-1)
    preds = (probs >= 0.5).astype(float)
    return {
        "type": "noul",
        "accuracy": float((preds == labels).mean()),
        "brier": brier_score(probs, labels),
        "ece": ece(probs, labels),
    }


def categorical_metrics(probs: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=int).reshape(-1)
    preds = probs.argmax(axis=-1)
    expected = (probs * np.arange(probs.shape[1])).sum(axis=1)
    return {
        "type": "categorical",
        "accuracy": float((preds == labels).mean()),
        "mean_expected_label": float(expected.mean()),
    }


def _index_map(spec: QuestionSpec) -> dict[str, int]:
    if isinstance(spec, ChoiceSpec):
        return {opt: i for i, opt in enumerate(spec.options)}
    if isinstance(spec, ScoreSpec):
        return {lvl: i for i, lvl in enumerate(spec.levels)}
    return {}


def _label_indices(spec: QuestionSpec, labels: list[Any]) -> list[int]:
    if isinstance(spec, ScoreSpec):
        return [int(v) for v in labels]
    mapping = _index_map(spec)
    return [mapping[v] for v in labels]


def _build_targets(spec: QuestionSpec, labels: list[Any]) -> torch.Tensor:
    if isinstance(spec, NoulSpec):
        return torch.tensor([float(v) for v in labels])
    return torch.tensor(_label_indices(spec, labels), dtype=torch.long)


def _evaluate_head(
    head, features: torch.Tensor, spec, labels: list[Any]
) -> dict[str, float]:
    with torch.no_grad():
        logits = head.logits(features)
        if isinstance(spec, NoulSpec):
            probs = torch.sigmoid(logits).numpy()
            return noul_metrics(probs, np.asarray(labels, dtype=float))
        probs = torch.softmax(logits, dim=-1).numpy()
        indices = np.asarray(_label_indices(spec, labels), dtype=int)
        metrics = categorical_metrics(probs, indices)
        metrics["type"] = "choice" if isinstance(spec, ChoiceSpec) else "score"
        return metrics


def encode_examples(
    specs: dict[str, QuestionSpec],
    examples: list[dict[str, Any]],
    encoder: TextEncoder,
) -> dict[str, torch.Tensor]:
    """Признаки по каждому вопросу: [n_examples, dim]."""
    states = [ex["state"] for ex in examples]
    return {
        name: encoder.encode([build_prompt(state, spec) for state in states])
        for name, spec in specs.items()
    }


def _metrics_report(
    specs: dict[str, QuestionSpec],
    examples: list[dict[str, Any]],
    per_question: dict[str, torch.Tensor],
    heads: DecisionHeads,
) -> dict[str, Any]:
    question_metrics: dict[str, dict[str, float]] = {}
    for name, spec in specs.items():
        labels = [ex["labels"][name] for ex in examples]
        question_metrics[name] = _evaluate_head(
            heads.heads[name], per_question[name], spec, labels
        )
    overall = float(np.mean([m["accuracy"] for m in question_metrics.values()]))
    return {"questions": question_metrics, "overall_accuracy": overall}


def train_benchmark(
    specs: dict[str, QuestionSpec],
    examples: list[dict[str, Any]],
    encoder: TextEncoder,
    *,
    train_steps: int = 0,
    lr: float = 0.1,
    seed: int = 0,
) -> tuple[DecisionHeads, dict[str, Any]]:
    """Обучает головы на ``examples`` и возвращает их вместе с отчётом на них же."""
    torch.manual_seed(seed)
    per_question = encode_examples(specs, examples, encoder)
    dim = next(iter(per_question.values())).shape[-1]
    heads = DecisionHeads(dim, specs, seed=seed)

    if train_steps:
        target_map = {
            name: _build_targets(spec, [ex["labels"][name] for ex in examples])
            for name, spec in specs.items()
        }
        SFTTrainer(heads, lr=lr).fit(
            features=per_question, targets=target_map, steps=train_steps, batch_size=32
        )

    report = _metrics_report(specs, examples, per_question, heads)
    report.update(
        n=len(examples),
        encoder=type(encoder).__name__,
        train_steps=train_steps,
    )
    return heads, report


def evaluate_benchmark(
    specs: dict[str, QuestionSpec],
    examples: list[dict[str, Any]],
    encoder: TextEncoder,
    heads: DecisionHeads,
) -> dict[str, Any]:
    """Метрики готовых голов на произвольном (например, отложенном) наборе."""
    per_question = encode_examples(specs, examples, encoder)
    report = _metrics_report(specs, examples, per_question, heads)
    report["n"] = len(examples)
    return report


def run_benchmark(
    specs: dict[str, QuestionSpec],
    examples: list[dict[str, Any]],
    encoder: TextEncoder,
    *,
    train_steps: int = 0,
    lr: float = 0.1,
    seed: int = 0,
    splits: dict[str, list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Единый прогон. Без ``splits`` — train/eval на одних примерах; со сплитами
    обучает на ``splits["train"]`` и меряет обобщение на ``val``/``test``.
    """
    if splits is None:
        _, report = train_benchmark(
            specs, examples, encoder, train_steps=train_steps, lr=lr, seed=seed
        )
        return report

    heads, train_report = train_benchmark(
        specs, splits["train"], encoder, train_steps=train_steps, lr=lr, seed=seed
    )
    split_reports: dict[str, dict[str, Any]] = {"train": train_report}
    for name, part in splits.items():
        if name == "train":
            continue
        split_reports[name] = evaluate_benchmark(specs, part, encoder, heads)

    primary_name = "test" if "test" in split_reports else "val"
    primary = split_reports[primary_name]
    report = dict(primary)
    report.update(
        encoder=type(encoder).__name__,
        train_steps=train_steps,
        splits=split_reports,
    )
    return report
