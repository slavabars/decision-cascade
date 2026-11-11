"""Обучение головы лабиринта штатными средствами проекта.

Никакого отдельного тренера: `eval.benchmark.train_benchmark` и
`training.recipe.calibrate_temperature` — те же функции, что у остальных наборов.
Температура подбирается на val и применяется только если ECE и Brier не стали
хуже (тот же гейт «не навреди»).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from api.server.service import HashStateEncoder
from eval.benchmark.core import evaluate_benchmark, train_benchmark
from eval.calibration.metrics import brier_score, ece
from eval.datasets.text_dataset import split_examples
from examples.games.maze.specs import MAZE_SPECS
from src.sampler.questions import build_prompt, questions_signature
from training.recipe.calibrate_temperature import (
    accept_calibration,
    fit_temperature,
)


def _probabilities(heads, encoder, examples: list[dict]) -> tuple[torch.Tensor, torch.Tensor]:
    prompts = [
        build_prompt(example["state"], MAZE_SPECS["need_deep"]) for example in examples
    ]
    with torch.no_grad():
        logits = heads.heads["need_deep"].logits(encoder.encode(prompts))
    probs = torch.sigmoid(logits)
    targets = torch.tensor(
        [float(example["labels"]["need_deep"]) for example in examples]
    )
    return probs, targets


def _report(heads, encoder, examples: list[dict]) -> dict[str, float]:
    probs, targets = _probabilities(heads, encoder, examples)
    p = probs.numpy()
    t = targets.numpy()
    return {
        "brier": float(brier_score(p, t)),
        "ece": float(ece(p, t, n_bins=10)),
        "accuracy": float(((p >= 0.5).astype(float) == t).mean()),
    }


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/maze/maze.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("checkpoints/packs/maze"))
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    examples = [
        json.loads(line)
        for line in args.data.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    splits = split_examples(examples, val_fraction=0.2, test_fraction=0.2, seed=args.seed)
    encoder = HashStateEncoder(args.dim)
    heads, _train_report = train_benchmark(
        MAZE_SPECS,
        splits["train"],
        encoder,
        train_steps=args.steps,
        lr=args.lr,
        seed=args.seed,
        rlcd_gate_examples=splits["val"],
    )

    before = _report(heads, encoder, splits["val"])
    prompts = [
        build_prompt(example["state"], MAZE_SPECS["need_deep"])
        for example in splits["val"]
    ]
    with torch.no_grad():
        logits = heads.heads["need_deep"].logits(encoder.encode(prompts))
    targets = torch.tensor(
        [float(example["labels"]["need_deep"]) for example in splits["val"]]
    )
    temperature = fit_temperature(logits, targets, noul=True)
    heads.set_temperatures({"need_deep": temperature})
    after = _report(heads, encoder, splits["val"])
    if not accept_calibration(before, after):
        heads.set_temperatures({"need_deep": 1.0})
        temperature = 1.0
        after = before

    metrics = {}
    for name in ("val", "test"):
        report = evaluate_benchmark(MAZE_SPECS, splits[name], encoder, heads)
        calibrated = _report(heads, encoder, splits[name])
        metrics[name] = {
            "overall_accuracy": report["overall_accuracy"],
            "calibration": calibrated,
            "n": len(splits[name]),
        }

    args.out.mkdir(parents=True, exist_ok=True)
    heads.save_checkpoint(args.out / "heads.pt")
    signature = questions_signature(MAZE_SPECS)
    heads.save_checkpoint(args.out / f"{signature}.pt")
    metadata = {
        "schema_version": "1",
        "encoder": "hash",
        "feature_dim": args.dim,
        "specs": "maze",
        "dataset_name": "maze",
        "questions_signature": signature,
        "splits": {name: len(part) for name, part in splits.items()},
        "temperature": {"need_deep": temperature},
        "metrics": metrics,
        "train_steps": args.steps,
        "seed": args.seed,
    }
    (args.out / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(
        f"temp={temperature:.2f} | val: brier {before['brier']:.4f} -> "
        f"{after['brier']:.4f}, ece {before['ece']:.4f} -> {after['ece']:.4f}"
    )
    for name in ("val", "test"):
        item = metrics[name]
        print(
            f"  {name:<4} acc={item['overall_accuracy']:.3f} "
            f"brier={item['calibration']['brier']:.4f} "
            f"ece={item['calibration']['ece']:.4f} (n={item['n']})"
        )
    print(f"пак -> {args.out}")
    return metadata


if __name__ == "__main__":
    main()
