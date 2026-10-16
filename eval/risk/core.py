"""Честное сравнение маршрутизации: локально / порог уверенности / бюджет риска.

Считает на одном наборе качество локальных решений и компромисс «эскалация vs
качество принятых» для двух каскадов:

- пороговый — эскалация при ``confidence < threshold`` (текущее поведение);
- risk-aware — эскалация при ``P(ошибка) > risk_budget``.

Падение (fallback) считается идеальным: так измеряется *качество маршрутизации*,
а не сила внешней модели. Дополнительно считается AURC/eAURC — качество
ранжирования «быть правильным» по уверенности и по оценке риска.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch

from eval.calibration.selective import aurc, coverage_at_risk, eaurc, risk_at_coverage
from src.sampler.questions import ChoiceSpec, NoulSpec, QuestionSpec
from src.schemas import DecisionResult


def labels_for(
    spec: QuestionSpec, examples: list[dict[str, Any]], name: str
) -> np.ndarray:
    raw = [ex["labels"][name] for ex in examples]
    if isinstance(spec, NoulSpec):
        return np.asarray(raw, dtype=float)
    if isinstance(spec, ChoiceSpec):
        index = {opt: i for i, opt in enumerate(spec.options)}
        return np.asarray([index[v] for v in raw], dtype=int)
    return np.asarray([int(v) for v in raw], dtype=int)


def is_correct(spec: QuestionSpec, decision: DecisionResult, label: Any) -> bool:
    if isinstance(spec, NoulSpec):
        return (decision.noul >= 0.5) == (float(label) >= 0.5)
    if isinstance(spec, ChoiceSpec):
        # ответ задан выбором: либо сам `choice`, либо индекс аргмакса распределения
        try:
            predicted = list(spec.options).index(decision.choice)
        except ValueError:
            probs = list(decision.probabilities.values())
            predicted = max(range(len(probs)), key=probs.__getitem__)
        return predicted == int(label)
    # ScoreSpec: как и в categorical_metrics — предсказанный уровень (argmax)
    probs = list(decision.probabilities.values())
    predicted = max(range(len(probs)), key=probs.__getitem__)
    return predicted == int(label)


def decisions_for_examples(
    heads, per_question: dict[str, torch.Tensor], n_examples: int
) -> dict[str, list[DecisionResult]]:
    """Решения по каждому вопросу для каждого примера — через ту же логику, что в сервисе."""
    names = list(heads.questions)
    collected: dict[str, list[DecisionResult]] = {name: [] for name in names}
    for i in range(n_examples):
        row = torch.stack([per_question[name][i] for name in names])
        results = heads.evaluate(row)
        for name in names:
            collected[name].append(results[name])
    return collected


def flatten(
    specs: dict[str, QuestionSpec],
    decisions: dict[str, list[DecisionResult]],
    examples: list[dict[str, Any]],
) -> tuple[list[DecisionResult], np.ndarray]:
    """Плоский список решений и их корректности по всем вопросам."""
    flat_decisions: list[DecisionResult] = []
    flat_correct: list[float] = []
    for name, spec in specs.items():
        labels = labels_for(spec, examples, name)
        for decision, label in zip(decisions[name], labels):
            flat_decisions.append(decision)
            flat_correct.append(float(is_correct(spec, decision, label)))
    return flat_decisions, np.asarray(flat_correct, dtype=float)


def threshold_escalation(decisions: list[DecisionResult], threshold: float) -> np.ndarray:
    return np.asarray([d.confidence < threshold for d in decisions], dtype=bool)


def risk_escalation(decisions: list[DecisionResult], model, budget: float) -> np.ndarray:
    return np.asarray(
        [model.should_escalate(d, budget) for d in decisions], dtype=bool
    )


def cascade_summary(correct: np.ndarray, escalate: np.ndarray) -> dict[str, float]:
    """Метрики каскада при идеальном fallback."""
    correct = np.asarray(correct, dtype=float).reshape(-1)
    escalate = np.asarray(escalate, dtype=bool).reshape(-1)
    n = correct.size
    accepted = ~escalate
    accepted_accuracy = float(correct[accepted].mean()) if accepted.any() else float("nan")
    end2end = float((correct[accepted].sum() + escalate.sum()) / n)
    return {
        "escalation_rate": float(escalate.mean()),
        "accepted_rate": float(accepted.mean()),
        "accepted_accuracy": accepted_accuracy,
        "accepted_risk": 1.0 - accepted_accuracy if accepted.any() else float("nan"),
        "end2end_accuracy": end2end,
    }


def ranker_report(scores: np.ndarray, correct: np.ndarray) -> dict[str, float]:
    """AURC/eAURC и покрытие при бюджете риска 5% и 10%."""
    return {
        "aurc": aurc(scores, correct),
        "eaurc": eaurc(scores, correct),
        "coverage_at_5pct_risk": coverage_at_risk(scores, correct, 0.05),
        "coverage_at_10pct_risk": coverage_at_risk(scores, correct, 0.10),
    }


DEFAULT_COVERAGES = (0.5, 0.7, 0.9)


def selective_accuracy(
    scores: np.ndarray, correct: np.ndarray, coverages=DEFAULT_COVERAGES
) -> dict[str, float]:
    """Точность среди принимаемых решений при заданной доле принятия.

    Честное сравнение при *равных затратах*: какой подмножество принимаемых
    решений точнее, если во внешнюю модель уходит одинаковая доля запросов.
    """
    return {
        f"{c:.1f}": 1.0 - risk_at_coverage(scores, correct, c)
        for c in coverages
    }
