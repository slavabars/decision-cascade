"""Честный отчёт: необученные vs обученные головы + калибровка на реальном наборе.

Считает accuracy/Brier/ECE по каждому вопросу до и после обучения, сохраняет
``metrics.json`` и reliability-диаграммы (PNG). По умолчанию — публичный AG News;
для синтетических тикетов задайте ``--specs support``.

Пример:
    python -m eval.report.run --specs public --encoder hash --dim 64 \
        --train-steps 300 --limit 4000 --out-dir data/report_ag_news
    python -m eval.report.run --specs public --encoder openai-community/gpt2 \
        --dim 768 --train-steps 500 --device cuda --out-dir data/report_ag_news_gpt2
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from api.server.service import HashStateEncoder
from eval.benchmark.core import encode_examples, evaluate_benchmark, train_benchmark
from eval.datasets.text_dataset import QUESTIONS as SUPPORT_QUESTIONS
from eval.datasets.text_dataset import (
    generate_examples,
    spec_from_api,
    split_examples,
)
from src.sampler.questions import ChoiceSpec, NoulSpec, QuestionSpec


def _encoder(name: str, dim: int, device: str | None, dtype: str | None):
    if name == "hash":
        return HashStateEncoder(dim)
    from src.device import check_dtype_device, resolve_device, resolve_dtype
    from src.encoder.factory import create_state_encoder

    resolved_device = resolve_device(device)
    resolved_dtype = resolve_dtype(dtype)
    check_dtype_device(resolved_dtype, resolved_device)
    raw_max = os.environ.get("DECISION_MAX_LENGTH", "256").strip().lower()
    max_length = None if raw_max in ("", "none", "0") else int(raw_max)
    return create_state_encoder(
        name,
        dim,
        dtype=resolved_dtype,
        device=resolved_device,
        max_length=max_length,
        batch_size=int(os.environ.get("DECISION_ENCODE_BATCH", "16")),
    ).load()


def _specs(name: str) -> dict[str, QuestionSpec]:
    if name == "support":
        source = SUPPORT_QUESTIONS
    elif name == "routerbench":
        from eval.datasets.routerbench import ROUTERBENCH_QUESTIONS

        source = ROUTERBENCH_QUESTIONS
    else:
        source = _public_questions()
    return {key: spec_from_api(data) for key, data in source.items()}


def _public_questions() -> dict[str, dict[str, Any]]:
    from eval.datasets.public_dataset import PUBLIC_QUESTIONS

    return PUBLIC_QUESTIONS


def _load_splits(args) -> dict[str, list[dict[str, Any]]]:
    if args.dataset is not None:
        base = Path(args.dataset)
        splits = {}
        for split in ("train", "test"):
            path = base / f"{split}.jsonl" if base.is_dir() else base
            with path.open(encoding="utf-8") as fh:
                splits[split] = [json.loads(line) for line in fh if line.strip()]
        return splits
    if args.specs == "public":
        from eval.datasets.public_dataset import load_ag_news

        splits = load_ag_news(args.limit or None)
        carved = split_examples(
            splits["train"], val_fraction=0.1, test_fraction=0.0, seed=args.seed
        )
        carved["test"] = splits["test"]
        return carved
    if args.specs == "routerbench":
        from eval.datasets.routerbench import load_routerbench

        splits = load_routerbench(args.limit or None, shot=args.shot, seed=args.seed)
        carved = split_examples(
            splits["train"], val_fraction=0.1, test_fraction=0.0, seed=args.seed
        )
        carved["test"] = splits["test"]
        return carved
    examples = generate_examples(n=args.limit or 1000, seed=args.seed)
    return split_examples(
        examples, val_fraction=0.2, test_fraction=0.2, seed=args.seed
    )


def _labels(spec: QuestionSpec, examples: list[dict[str, Any]], name: str) -> np.ndarray:
    raw = [ex["labels"][name] for ex in examples]
    if isinstance(spec, NoulSpec):
        return np.asarray(raw, dtype=float)
    if isinstance(spec, ChoiceSpec):
        index = {opt: i for i, opt in enumerate(spec.options)}
        return np.asarray([index[v] for v in raw], dtype=int)
    return np.asarray([int(v) for v in raw], dtype=int)


def _confidence_and_correct(
    spec: QuestionSpec, head, features: torch.Tensor, labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    with torch.no_grad():
        logits = head.logits(features)
        if isinstance(spec, NoulSpec):
            probs = torch.sigmoid(logits).numpy()
            return probs, (probs >= 0.5).astype(float)
        probs = torch.softmax(logits, dim=-1).numpy()
        return probs.max(axis=-1), (probs.argmax(axis=-1) == labels).astype(float)


def _save_diagrams(specs, heads, test_examples, encoder, out_dir: Path) -> list[str]:
    import matplotlib.pyplot as plt

    from eval.calibration.metrics import calibration_curve

    per_question = encode_examples(specs, test_examples, encoder)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, spec in specs.items():
        labels = _labels(spec, test_examples, name)
        confidence, correct = _confidence_and_correct(
            spec, heads.heads[name], per_question[name], labels
        )
        fig = calibration_curve(confidence, correct, n_bins=10)
        kind = "noul" if isinstance(spec, NoulSpec) else "categorical"
        fig.suptitle(f"{name} ({kind})")
        path = out_dir / f"reliability_{name}.png"
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        written.append(path.name)
    return written


def _print_compare(
    specs, untrained: dict, trained: dict
) -> dict[str, dict[str, float]]:
    print(f"{'question':<12} {'type':<6} {'acc(off)':>9} {'acc(on)':>9} "
          f"{'ece(off)':>9} {'ece(on)':>9}")
    summary = {}
    for name, spec in specs.items():
        if isinstance(spec, NoulSpec):
            kind = "noul"
        elif isinstance(spec, ChoiceSpec):
            kind = "choice"
        else:
            kind = "score"
        off = untrained["questions"][name]
        on = trained["questions"][name]
        print(
            f"{name:<12} {kind:<6} {off['accuracy']:>9.3f} {on['accuracy']:>9.3f} "
            f"{off.get('ece', float('nan')):>9.3f} {on.get('ece', float('nan')):>9.3f}"
        )
        summary[name] = {
            "type": kind,
            "untrained": off,
            "trained": on,
        }
    print(
        f"{'OVERALL':<12} {'':<6} {untrained['overall_accuracy']:>9.3f} "
        f"{trained['overall_accuracy']:>9.3f}"
    )
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Untrained vs trained report")
    parser.add_argument(
        "--specs", choices=["support", "public", "routerbench"], default="public"
    )
    parser.add_argument("--dataset", type=Path, default=None, help="каталог train/test jsonl")
    parser.add_argument("--encoder", default="hash", help="'hash' или model_id")
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--train-steps", type=int, default=300)
    parser.add_argument("--rlcd-steps", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument(
        "--choice-head",
        choices=["linear", "option_scoring"],
        default="linear",
        help="голова выбора: отдельный вес на вариант или общий скорер по эмбеддингам",
    )
    parser.add_argument("--limit", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--shot", choices=["0shot", "5shot"], default="0shot", help="RouterBench"
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--out-dir", type=Path, default=Path("data/report"))
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)

    specs = _specs(args.specs)
    splits = _load_splits(args)
    train, test = splits["train"], splits["test"]
    print(
        f"specs={args.specs} encoder={args.encoder} train={len(train)} test={len(test)}"
    )

    encoder = _encoder(args.encoder, args.dim, args.device, args.dtype)
    untrained_heads, _ = train_benchmark(
        specs, train, encoder, train_steps=0, seed=args.seed,
        choice_head=args.choice_head,
    )
    untrained_test = evaluate_benchmark(specs, test, encoder, untrained_heads)
    heads, _ = train_benchmark(
        specs,
        train,
        encoder,
        train_steps=args.train_steps,
        lr=args.lr,
        seed=args.seed,
        rlcd_steps=args.rlcd_steps,
        choice_head=args.choice_head,
    )
    trained_test = evaluate_benchmark(specs, test, encoder, heads)

    summary = _print_compare(specs, untrained_test, trained_test)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "specs": args.specs,
        "encoder": args.encoder,
        "feature_dim": args.dim,
        "train_steps": args.train_steps,
        "rlcd_steps": args.rlcd_steps,
        "choice_head": args.choice_head,
        "n_train": len(train),
        "n_test": len(test),
        "questions": summary,
        "untrained_overall_accuracy": untrained_test["overall_accuracy"],
        "trained_overall_accuracy": trained_test["overall_accuracy"],
    }
    (args.out_dir / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"metrics -> {args.out_dir / 'metrics.json'}")

    if not args.no_plots:
        names = _save_diagrams(specs, heads, test, encoder, args.out_dir)
        print(f"diagrams-> {', '.join(names)}")


if __name__ == "__main__":
    main()
