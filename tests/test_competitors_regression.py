"""Регрессия конвейера бенчмарка конкурентов на детерминированном mock-сервисе.

Числа mock-конкурента зависят только от (state, вопрос, seed) и генератора
датасета, поэтому они воспроизводимы до бита. Если конвейер оценки, протокол
или генератор метрик изменятся, тест упадёт — это и есть CI-регрессия.
"""
from __future__ import annotations

import pytest

from eval.competitors.evaluate import evaluate_backend
from eval.competitors.mocks import mock_systemone_transport
from eval.datasets.text_dataset import QUESTIONS, generate_examples, spec_from_api
from src.backends.http import HttpSystemOneBackend

# Эталон: support-набор, seed=7, n=64, mock seed=0 (см. tests/test_competitors.py).
REFERENCE_OVERALL = 0.289062
REFERENCE_QUESTIONS = {
    "срочность": {"accuracy": 0.515625, "brier": 0.284443, "ece": 0.236753},
    "команда": {"accuracy": 0.21875, "brier": 0.902806, "ece": 0.242511},
    "причина": {"accuracy": 0.1875, "brier": 0.811222, "ece": 0.255558},
    "накал": {"accuracy": 0.25, "brier": 0.823816, "ece": 0.122802},
    "эскалация": {"accuracy": 0.390625, "brier": 0.410094, "ece": 0.382136},
    "удержание": {"accuracy": 0.171875, "brier": 0.899853, "ece": 0.196799},
}


def test_mock_competitor_metrics_are_stable():
    specs = {name: spec_from_api(data) for name, data in QUESTIONS.items()}
    examples = generate_examples(n=64, seed=7)
    backend = HttpSystemOneBackend(
        "http://mock", model="mock", transport=mock_systemone_transport(seed=0)
    )
    try:
        report = evaluate_backend(
            specs, examples, backend.decide, name="mock", warmup=0
        )
    finally:
        backend.close()

    assert report["n"] == 64
    assert report["overall_accuracy"] == pytest.approx(REFERENCE_OVERALL, abs=1e-6)
    assert report["escalated_decisions"] == 0
    for name, expected in REFERENCE_QUESTIONS.items():
        metrics = report["questions"][name]
        assert metrics["accuracy"] == pytest.approx(expected["accuracy"], abs=1e-6)
        assert metrics["brier"] == pytest.approx(expected["brier"], abs=1e-6)
        assert metrics["ece"] == pytest.approx(expected["ece"], abs=1e-6)
