"""Тесты офлайн-бенчмарка голов на сгенерированном датасете — RED до реализации."""
import numpy as np

from api.server.service import HashStateEncoder
from eval.benchmark.core import (
    categorical_metrics,
    noul_metrics,
    run_benchmark,
)
from eval.datasets.text_dataset import QUESTIONS, SPECS, generate_examples


def _encoder(dim=64):
    return HashStateEncoder(dim)


def test_noul_metrics_perfect_prediction():
    probs = np.array([0.9, 0.1, 0.8, 0.2])
    labels = np.array([1, 0, 1, 0])
    metrics = noul_metrics(probs, labels)
    assert metrics["accuracy"] == 1.0
    assert 0.0 <= metrics["brier"] <= 1.0
    assert 0.0 <= metrics["ece"] <= 1.0


def test_categorical_metrics_accuracy_and_mae():
    probs = np.array([[0.7, 0.2, 0.1], [0.1, 0.1, 0.8]])
    labels = np.array([0, 2])
    metrics = categorical_metrics(probs, labels)
    assert metrics["accuracy"] == 1.0
    assert abs(metrics["mean_expected_label"] - 1.05) < 1e-6


def test_run_benchmark_report_structure():
    examples = generate_examples(n=24, seed=0)
    report = run_benchmark(
        SPECS, examples, _encoder(), train_steps=0, seed=0
    )
    assert report["n"] == 24
    assert report["train_steps"] == 0
    assert set(report["questions"]) == set(QUESTIONS)
    for metrics in report["questions"].values():
        assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= report["overall_accuracy"] <= 1.0


def test_run_benchmark_is_deterministic():
    examples = generate_examples(n=16, seed=5)
    a = run_benchmark(SPECS, examples, _encoder(), train_steps=5, seed=0)
    b = run_benchmark(SPECS, examples, _encoder(), train_steps=5, seed=0)
    assert a == b


def test_run_benchmark_training_runs_and_improves_overall():
    examples = generate_examples(n=200, seed=0)
    untrained = run_benchmark(SPECS, examples, _encoder(), train_steps=0, seed=0)
    trained = run_benchmark(SPECS, examples, _encoder(), train_steps=100, seed=0)
    assert trained["train_steps"] == 100
    assert trained["overall_accuracy"] >= untrained["overall_accuracy"]
