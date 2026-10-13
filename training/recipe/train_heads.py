"""Рецепт обучения голов решений на синтетике с честным train/val/test.

Обучает головы только на train-сплите, измеряет обобщение на val/test и
сохраняет чекпоинт + метаданные (энкодер, размерность, хэш датасета, метрики).
Веса публикуются как артефакт GitHub Release (см. ``publish_release.sh``), а не
в git.

Пример:
    python -m training.recipe.train_heads --generate 1000 \
        --out-dir checkpoints/support --encoder hash --dim 64 --train-steps 300
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from api.server.service import HashStateEncoder
from eval.benchmark.core import evaluate_benchmark, train_benchmark
from eval.datasets.text_dataset import (
    QUESTIONS,
    generate_examples,
    spec_from_api,
    split_examples,
    write_jsonl,
)
from src.sampler.questions import spec_to_dict
from src.schemas import SCHEMA_VERSION


def _load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _encode_batch() -> int:
    return int(os.environ.get("DECISION_ENCODE_BATCH", "16"))


def _make_encoder(name: str, dim: int):
    if name == "hash":
        return HashStateEncoder(dim)
    from src.encoder.gemma import GemmaStateEncoder

    return GemmaStateEncoder(
        model_id=name,
        feature_dim=dim,
        max_length=_max_length(),
        batch_size=_encode_batch(),
    ).load()


def _max_length() -> int | None:
    raw = os.environ.get("DECISION_MAX_LENGTH", "256").strip().lower()
    if raw in ("", "none", "0"):
        return None
    return int(raw)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train decision heads (recipe)")
    parser.add_argument("--dataset", type=Path, default=None, help="JSONL датасет")
    parser.add_argument(
        "--generate", type=int, default=0, help="сгенерировать N примеров"
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--encoder", default="hash", help="'hash' или model_id")
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--train-steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> dict:
    args = _parser().parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset is not None:
        dataset_path = Path(args.dataset)
        examples = _load_jsonl(dataset_path)
    else:
        n = args.generate or 200
        examples = generate_examples(n=n, seed=args.seed)
        dataset_path = out_dir / "dataset.jsonl"
        write_jsonl(dataset_path, examples)

    dataset_hash = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    specs = {name: spec_from_api(data) for name, data in QUESTIONS.items()}
    splits = split_examples(
        examples,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )

    encoder = _make_encoder(args.encoder, args.dim)
    heads, train_report = train_benchmark(
        specs,
        splits["train"],
        encoder,
        train_steps=args.train_steps,
        lr=args.lr,
        seed=args.seed,
    )
    metrics = {"train": train_report}
    for name in ("val", "test"):
        metrics[name] = evaluate_benchmark(specs, splits[name], encoder, heads)

    checkpoint = heads.save_checkpoint(out_dir / "heads.pt")
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "encoder": args.encoder,
        "feature_dim": heads.dim,
        "max_length": _max_length(),
        "dataset": {
            "path": str(dataset_path),
            "sha256": dataset_hash,
            "n": len(examples),
        },
        "splits": {name: len(part) for name, part in splits.items()},
        "metrics": {
            name: {
                "overall_accuracy": report["overall_accuracy"],
                "questions": report["questions"],
            }
            for name, report in metrics.items()
        },
        "questions": {name: spec_to_dict(spec) for name, spec in specs.items()},
        "train_steps": args.train_steps,
        "lr": args.lr,
        "seed": args.seed,
    }
    meta_path = out_dir / "metadata.json"
    meta_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(f"checkpoint -> {checkpoint}")
    print(f"metadata   -> {meta_path}")
    for name in ("train", "val", "test"):
        acc = metadata["metrics"][name]["overall_accuracy"]
        print(f"  {name:<5} acc={acc:.3f} (n={metadata['splits'][name]})")
    return metadata


if __name__ == "__main__":
    main()
