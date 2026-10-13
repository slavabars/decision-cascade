"""CLI: офлайн-бенчмарк голов решений на датасете.

Пример:
    python -m eval.benchmark.run --dataset data/support_tickets.jsonl \
        --encoder hash --train-steps 200 --out data/report.json
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from api.server.service import HashStateEncoder
from eval.benchmark.core import run_benchmark
from eval.datasets.text_dataset import QUESTIONS, spec_from_api, split_examples


def _load_examples(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _max_length() -> int | None:
    raw = os.environ.get("DECISION_MAX_LENGTH", "256").strip().lower()
    if raw in ("", "none", "0"):
        return None
    return int(raw)


def _encode_batch() -> int:
    return int(os.environ.get("DECISION_ENCODE_BATCH", "16"))


def _make_encoder(name: str, dim: int):
    if name == "hash":
        return HashStateEncoder(dim)
    from src.encoder.factory import create_state_encoder

    return create_state_encoder(
        name,
        dim,
        max_length=_max_length(),
        batch_size=_encode_batch(),
    ).load()


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline decision benchmark")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--encoder", default="hash", help="'hash' или model_id")
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--train-steps", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0, help="0 = все примеры")
    parser.add_argument("--val-fraction", type=float, default=0.0)
    parser.add_argument("--test-fraction", type=float, default=0.0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    examples = _load_examples(args.dataset)
    if args.limit:
        examples = examples[: args.limit]
    specs = {name: spec_from_api(data) for name, data in QUESTIONS.items()}

    splits = None
    if args.val_fraction or args.test_fraction:
        splits = split_examples(
            examples,
            val_fraction=args.val_fraction,
            test_fraction=args.test_fraction,
            seed=args.seed,
        )

    encoder = _make_encoder(args.encoder, args.dim)
    report = run_benchmark(
        specs,
        examples,
        encoder,
        train_steps=args.train_steps,
        lr=args.lr,
        seed=args.seed,
        splits=splits,
    )

    print(f"n={report['n']} encoder={report['encoder']} train_steps={report['train_steps']}")
    for name, metrics in report["questions"].items():
        extra = ""
        if "brier" in metrics:
            extra = f" brier={metrics['brier']:.3f} ece={metrics['ece']:.3f}"
        elif "mean_expected_label" in metrics:
            extra = f" mean_label={metrics['mean_expected_label']:.2f}"
        print(f"  {name:<12} [{metrics['type']:<5}] acc={metrics['accuracy']:.3f}{extra}")
    print(f"overall_accuracy={report['overall_accuracy']:.3f}")
    for split_name, split_report in report.get("splits", {}).items():
        print(f"  [{split_name}] acc={split_report['overall_accuracy']:.3f}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"report -> {args.out}")


if __name__ == "__main__":
    main()
