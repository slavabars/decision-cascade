"""Оценка любого SystemOne-совместимого бэкенда: качество + латентность + расход.

Бэкенд — это либо наши ``LocalHeadsBackend``/``CascadeBackend``, либо внешний
``HttpSystemOneBackend`` (Clef/Clef-Flash/Laya/Perplexity и т.п.). Все меряются
на одном размеченном наборе и одном наборе вопросов, поэтому сравнение честное.
"""
from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import numpy as np

from eval.benchmark.core import categorical_metrics, noul_metrics
from eval.latency.benchmark import summarize_latencies
from src.backends.base import BackendResult
from src.sampler.questions import ChoiceSpec, NoulSpec, QuestionSpec, ScoreSpec

Decide = Callable[[str, dict[str, QuestionSpec]], BackendResult]


def _option_keys(spec: QuestionSpec) -> list[str]:
    if isinstance(spec, ChoiceSpec):
        return list(spec.options)
    if isinstance(spec, ScoreSpec):
        return list(spec.levels)
    raise TypeError(f"question has no options/levels: {type(spec).__name__}")


def _truth_index(spec: QuestionSpec, value: Any) -> int:
    if isinstance(spec, ScoreSpec):
        return int(value)
    return _option_keys(spec).index(value)


def _flatten_usage(usages: list[dict[str, Any]]) -> dict[str, float]:
    """Складывает числовые поля usage (включая вложенные) в плоский словарь."""
    totals: dict[str, float] = {}

    def add(prefix: str, value: Any) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                add(f"{prefix}.{key}" if prefix else key, item)
        elif isinstance(value, bool):
            return
        elif isinstance(value, (int, float)):
            totals[prefix] = totals.get(prefix, 0.0) + float(value)

    for usage in usages:
        add("", usage)
    return totals


def evaluate_backend(
    specs: dict[str, QuestionSpec],
    examples: list[dict[str, Any]],
    decide: Decide,
    *,
    name: str,
    warmup: int = 2,
) -> dict[str, Any]:
    """Метрики качества, латентности и расхода одного бэкенда на наборе."""
    if examples and warmup:
        for _ in range(warmup):
            decide(examples[0]["state"], specs)

    noul_probs = {n: [] for n, s in specs.items() if isinstance(s, NoulSpec)}
    noul_true = {n: [] for n in noul_probs}
    cat_probs = {n: [] for n, s in specs.items() if not isinstance(s, NoulSpec)}
    cat_true = {n: [] for n in cat_probs}
    latencies: list[float] = []
    usages: list[dict[str, Any]] = []
    requests_with_escalation = 0

    for example in examples:
        start = time.perf_counter()
        result = decide(example["state"], specs)
        latencies.append((time.perf_counter() - start) * 1000.0)
        usages.append(dict(result.usage))
        if int(result.usage.get("escalated", 0)) > 0:
            requests_with_escalation += 1
        for qname, spec in specs.items():
            decision = result.decisions[qname]
            truth = example["labels"][qname]
            if isinstance(spec, NoulSpec):
                noul_probs[qname].append(float(decision.noul))
                noul_true[qname].append(float(truth))
            else:
                probs = decision.probabilities or {}
                cat_probs[qname].append(
                    [float(probs[key]) for key in _option_keys(spec)]
                )
                cat_true[qname].append(_truth_index(spec, truth))

    questions: dict[str, dict[str, float]] = {}
    for qname, spec in specs.items():
        if isinstance(spec, NoulSpec):
            questions[qname] = noul_metrics(
                np.asarray(noul_probs[qname]), np.asarray(noul_true[qname])
            )
        else:
            questions[qname] = categorical_metrics(
                np.asarray(cat_probs[qname]), np.asarray(cat_true[qname])
            )

    question_decisions = len(examples) * len(specs)
    escalated = int(sum(int(u.get("escalated", 0)) for u in usages))
    return {
        "name": name,
        "n": len(examples),
        "overall_accuracy": float(
            np.mean([m["accuracy"] for m in questions.values()])
        ),
        "questions": questions,
        "latency_ms": summarize_latencies(latencies) if latencies else {},
        "escalated_decisions": escalated,
        "escalation_rate": (
            escalated / question_decisions if question_decisions else 0.0
        ),
        "requests_with_escalation": requests_with_escalation,
        "usage_totals": _flatten_usage(usages),
    }
