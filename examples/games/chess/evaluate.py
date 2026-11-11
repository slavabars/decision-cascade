"""Оценка головы `need_deep`: качество против цены глубоких просчётов.

Отвечает на единственный вопрос, который важен для каскада: сколько глубоких
просчётов можно не делать, сохраняя качество. Считается на отложенном сплите,
который не участвовал ни в обучении, ни в подборе температуры.

    python -m examples.games.chess.evaluate --pack checkpoints/packs/chess \
        --data data/chess
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from eval.calibration.metrics import brier_score, ece
from eval.datasets.chess import CHESS_SPECS

THRESHOLDS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def auc(probs: np.ndarray, labels: np.ndarray) -> float:
    """ROC-AUC без sklearn: вероятность, что положительный пример выше по скору."""
    positive = probs[labels == 1]
    negative = probs[labels == 0]
    if len(positive) == 0 or len(negative) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([positive, negative]), kind="mergesort")
    ranks = np.empty(len(order), dtype=float)
    ranks[order] = np.arange(1, len(order) + 1)
    # средние ранги для совпадающих значений — иначе AUC смещается
    combined = np.concatenate([positive, negative])
    for value in np.unique(combined):
        mask = combined == value
        if mask.sum() > 1:
            ranks[mask] = ranks[mask].mean()
    rank_sum = ranks[: len(positive)].sum()
    n_pos, n_neg = len(positive), len(negative)
    return float((rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def agree_share(examples: list[dict]) -> float:
    """Как часто быстрый ход совпадает с глубоким (то есть «и так сойдёт»)."""
    if not examples:
        return float("nan")
    return sum(ex["shallow"].split()[0] == ex["deep"].split()[0] for ex in examples) / len(examples)


def quality_curve(
    probs: np.ndarray, examples: list[dict], *, shallow_cost: int, deep_cost: int
) -> list[dict]:
    """Качество и цена при разных порогах эскалации.

    Качество — доля позиций, где сыгранный ход совпал с глубоким ответом: при
    эскалации это гарантированно так, без неё — совпадает ли быстрый ход.
    Цена — среднее время на ход в мс.
    """
    agrees = np.array(
        [ex["shallow"].split()[0] == ex["deep"].split()[0] for ex in examples], dtype=float
    )
    rows = []
    for threshold in THRESHOLDS:
        escalated = probs >= threshold
        chosen_agrees = np.where(escalated, 1.0, agrees)
        rows.append(
            {
                "threshold": threshold,
                "escalation_rate": float(escalated.mean()),
                "quality": float(chosen_agrees.mean()),
                "avg_ms": float(
                    shallow_cost + escalated.mean() * deep_cost
                ),
            }
        )
    return rows


def _predict(pack: Path, examples: list[dict]) -> np.ndarray:
    from api.server.service import DecisionService, HashStateEncoder

    feature_dim = json.loads((pack / "metadata.json").read_text(encoding="utf-8"))[
        "feature_dim"
    ]
    service = DecisionService(
        encoder=HashStateEncoder(feature_dim),
        dim=feature_dim,
        checkpoint_dir=pack,
        untrained_policy="refuse",
    )
    answers = service.decide_many([ex["state"] for ex in examples], CHESS_SPECS)
    return np.array([float(answer["need_deep"].noul) for answer in answers])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, default=Path("checkpoints/packs/chess"))
    parser.add_argument("--data", type=Path, default=Path("data/chess"))
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--shallow-ms", type=int, default=60)
    parser.add_argument("--deep-ms", type=int, default=260)
    args = parser.parse_args(argv)

    path = args.data / f"{args.split}.jsonl"
    examples = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line
    ]
    probs = _predict(args.pack, examples)
    labels = np.array([ex["labels"]["need_deep"] for ex in examples])
    deep_cost = args.deep_ms - args.shallow_ms

    print(
        f"{args.split}: n={len(examples)} · need_deep={labels.mean():.2f} · "
        f"AUC={auc(probs, labels):.3f} · Brier={brier_score(probs, labels):.4f} · "
        f"ECE={ece(probs, labels, n_bins=10):.4f}"
    )
    print(f"быстрый ход совпадает с глубоким: {agree_share(examples):.2f}")
    print()
    print("| порог | эскалаций | качество | мс/ход |")
    print("| ---: | ---: | ---: | ---: |")
    for row in quality_curve(
        probs,
        examples,
        shallow_cost=args.shallow_ms,
        deep_cost=deep_cost,
    ):
        print(
            f"| {row['threshold']:.1f} | {row['escalation_rate']:.0%} | "
            f"{row['quality']:.3f} | {row['avg_ms']:.0f} |"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
