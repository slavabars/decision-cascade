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
from src.sampler.questions import questions_signature, spec_to_dict
from src.schemas import SCHEMA_VERSION


def _load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _encode_batch() -> int:
    return int(os.environ.get("DECISION_ENCODE_BATCH", "16"))


def _make_encoder(name: str, dim: int):
    if name == "hash":
        return HashStateEncoder(dim)
    from src.device import check_dtype_device, resolve_device, resolve_dtype
    from src.encoder.factory import create_state_encoder

    device = resolve_device(os.environ.get("DECISION_DEVICE"))
    dtype = resolve_dtype(os.environ.get("DECISION_DTYPE"))
    check_dtype_device(dtype, device)
    return create_state_encoder(
        name,
        dim,
        dtype=dtype,
        device=device,
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
        "--generate", type=int, default=0, help="сгенерировать N примеров (support)"
    )
    parser.add_argument(
        "--specs",
        choices=["support", "public", "routerbench"],
        default="support",
        help="набор вопросов: синтетические тикеты, публичный AG News или RouterBench",
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="для public: ограничение train-сплита"
    )
    parser.add_argument(
        "--shot",
        choices=["0shot", "5shot"],
        default="0shot",
        help="RouterBench: версия промптов (0-shot или 5-shot)",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--encoder", default="hash", help="'hash' или model_id")
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--train-steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument(
        "--choice-head",
        choices=["linear", "option_scoring"],
        default="linear",
        help="голова выбора: отдельный вес на вариант или общий скорер по эмбеддингам",
    )
    parser.add_argument(
        "--rlcd-steps",
        type=int,
        default=0,
        help="шаги RLCD-калибровки noul-голов после SFT (0 — выключено)",
    )
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def _load_data(
    args: argparse.Namespace, out_dir: Path
) -> tuple[dict, dict[str, list], Path, str]:
    """Готовит (спецификации, сплиты, путь датасета, имя датасета).

    ``support`` — синтетические тикеты (train/val/test режутся из одного набора),
    ``public`` — публичный AG News с настоящим train/test-сплитом,
    ``routerbench`` — RouterBench с детерминированным разбиением по ``seed``.
    """
    if args.specs == "public":
        from eval.datasets.public_dataset import PUBLIC_SPECS, load_ag_news

        base = load_ag_news(args.limit or None)
        carved = split_examples(
            base["train"],
            val_fraction=args.val_fraction,
            test_fraction=0.0,
            seed=args.seed,
        )
        splits = {
            "train": carved["train"],
            "val": carved["val"],
            "test": base["test"],
        }
        examples = splits["train"] + splits["test"]
        dataset_path = out_dir / "dataset.jsonl"
        write_jsonl(dataset_path, examples)
        return PUBLIC_SPECS, splits, dataset_path, "ag_news"

    if args.specs == "routerbench":
        from eval.datasets.routerbench import ROUTERBENCH_QUESTIONS, load_routerbench

        base = load_routerbench(args.limit or None, shot=args.shot)
        carved = split_examples(
            base["train"],
            val_fraction=args.val_fraction,
            test_fraction=0.0,
            seed=args.seed,
        )
        splits = {
            "train": carved["train"],
            "val": carved["val"],
            "test": base["test"],
        }
        examples = splits["train"] + splits["test"]
        dataset_path = out_dir / "dataset.jsonl"
        write_jsonl(dataset_path, examples)
        specs = {
            name: spec_from_api(data) for name, data in ROUTERBENCH_QUESTIONS.items()
        }
        return specs, splits, dataset_path, f"routerbench_{args.shot}"

    specs = {name: spec_from_api(data) for name, data in QUESTIONS.items()}
    if args.dataset is not None:
        dataset_path = Path(args.dataset)
        examples = _load_jsonl(dataset_path)
    else:
        examples = generate_examples(n=args.generate or 200, seed=args.seed)
        dataset_path = out_dir / "dataset.jsonl"
        write_jsonl(dataset_path, examples)
    splits = split_examples(
        examples,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )
    return specs, splits, dataset_path, "support"


def main(argv: list[str] | None = None) -> dict:
    args = _parser().parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    specs, splits, dataset_path, dataset_name = _load_data(args, out_dir)

    dataset_hash = hashlib.sha256(dataset_path.read_bytes()).hexdigest()

    encoder = _make_encoder(args.encoder, args.dim)
    heads, train_report = train_benchmark(
        specs,
        splits["train"],
        encoder,
        train_steps=args.train_steps,
        lr=args.lr,
        seed=args.seed,
        rlcd_steps=args.rlcd_steps,
        rlcd_gate_examples=splits["val"] or None,
        choice_head=args.choice_head,
    )
    metrics = {"train": train_report}
    for name in ("val", "test"):
        metrics[name] = evaluate_benchmark(specs, splits[name], encoder, heads)

    checkpoint = heads.save_checkpoint(out_dir / "heads.pt")
    # Копия под канонической подписью набора вопросов: сервис ищет чекпоинт как
    # ``<checkpoint_dir>/<questions_signature>.pt``, поэтому файл можно положить
    # в каталог DECISION_CHECKPOINT_DIR и он подхватится автоматически.
    signature = questions_signature(specs)
    signature_checkpoint = heads.save_checkpoint(out_dir / f"{signature}.pt")
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "encoder": args.encoder,
        "feature_dim": heads.dim,
        "max_length": _max_length(),
        "specs": args.specs,
        "dataset_name": dataset_name,
        "questions_signature": signature,
        "dataset": {
            "path": str(dataset_path),
            "sha256": dataset_hash,
            "n": sum(1 for line in dataset_path.open(encoding="utf-8") if line.strip()),
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
        "rlcd_steps": args.rlcd_steps,
        "lr": args.lr,
        "seed": args.seed,
    }
    meta_path = out_dir / "metadata.json"
    meta_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(f"checkpoint -> {checkpoint}")
    print(f"signature  -> {signature_checkpoint}")
    print(f"metadata   -> {meta_path}")
    for name in ("train", "val", "test"):
        acc = metadata["metrics"][name]["overall_accuracy"]
        print(f"  {name:<5} acc={acc:.3f} (n={metadata['splits'][name]})")
    return metadata


if __name__ == "__main__":
    main()
