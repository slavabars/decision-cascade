"""Метрики калибровки: ECE, Brier score, reliability/калибровочные кривые."""
from __future__ import annotations

import numpy as np


def ece(
    probs: np.ndarray,
    outcomes: np.ndarray,
    n_bins: int = 10,
    mode: str = "probability",
) -> float:
    """Expected Calibration Error: средневзвешенное |accuracy - confidence| по бинам."""
    del mode
    probs = np.asarray(probs, dtype=float)
    outcomes = np.asarray(outcomes, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = np.clip(np.digitize(probs, edges) - 1, 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        mask = bins == b
        count = int(mask.sum())
        if count == 0:
            continue
        conf = float(probs[mask].mean())
        acc = float(outcomes[mask].mean())
        total += (count / len(probs)) * abs(acc - conf)
    return total


def brier_score(probs: np.ndarray, outcomes: np.ndarray) -> float:
    probs = np.asarray(probs, dtype=float)
    outcomes = np.asarray(outcomes, dtype=float)
    return float(np.mean((probs - outcomes) ** 2))


def multiclass_brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    """Brier для многоклассовой задачи: mean_i sum_k (p_ik - 1[k == y_i])^2.

    Для бинарного случая эквивалентен ``brier_score`` (с точностью до множителя 2
    между «sum» и «mean» по классам), здесь используется стандартная сумма по классам.
    """
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=int).reshape(-1)
    if probs.ndim != 2:
        raise ValueError(f"probs must be [n, n_classes]; got {probs.shape}")
    if probs.shape[0] != labels.shape[0]:
        raise ValueError(
            f"probs rows {probs.shape[0]} != labels {labels.shape[0]}"
        )
    onehot = np.zeros_like(probs)
    onehot[np.arange(labels.shape[0]), labels] = 1.0
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def top_label_ece(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 10
) -> float:
    """ECE по уверенности лучшего класса: confidence = max_k p_k, outcome = top-1 верен."""
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=int).reshape(-1)
    if probs.ndim != 2:
        raise ValueError(f"probs must be [n, n_classes]; got {probs.shape}")
    confidence = probs.max(axis=-1)
    correct = (probs.argmax(axis=-1) == labels).astype(float)
    return ece(confidence, correct, n_bins=n_bins)


def reliability_curve(
    probs: np.ndarray, outcomes: np.ndarray, n_bins: int = 10
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    probs = np.asarray(probs, dtype=float)
    outcomes = np.asarray(outcomes, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = np.clip(np.digitize(probs, edges) - 1, 0, n_bins - 1)
    conf = np.zeros(n_bins)
    acc = np.zeros(n_bins)
    counts = np.zeros(n_bins, dtype=int)
    for b in range(n_bins):
        mask = bins == b
        n = int(mask.sum())
        counts[b] = n
        if n:
            conf[b] = float(probs[mask].mean())
            acc[b] = float(outcomes[mask].mean())
    return conf, acc, counts


def calibration_curve(
    probs: np.ndarray, outcomes: np.ndarray, n_bins: int = 10
):
    """Строит reliability diagram и возвращает matplotlib Figure."""
    import matplotlib.pyplot as plt

    conf, acc, counts = reliability_curve(probs, outcomes, n_bins=n_bins)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0.0, 1.0], [0.0, 1.0], "--", color="gray", label="perfect")
    nonempty = counts > 0
    ax.plot(conf[nonempty], acc[nonempty], "-", color="tab:blue", alpha=0.4)
    ax.scatter(
        conf[nonempty],
        acc[nonempty],
        s=counts[nonempty] / max(1, counts[nonempty].max()) * 300,
        color="tab:blue",
        label="model",
    )
    ax.set_xlabel("predicted probability")
    ax.set_ylabel("observed frequency")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.set_title("Reliability diagram")
    return fig