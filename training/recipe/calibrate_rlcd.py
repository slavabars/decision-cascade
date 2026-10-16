"""RLCD-калибровка noul-голов на реальных признаках энкодера (held-out гейт).

Загружает обученный пак голов (``--checkpoint-dir``) либо создаёт новые, считает
признаки энкодера на train/val, калибрует noul-головы RLCD и принимает изменение
только если оно улучшает калибровку на **val** (а не на train). Печатает Brier/ECE
до и после по каждому вопросу и среднюю абсолютную ошибку noul.

Пример:
    python -m training.recipe.calibrate_rlcd --specs public \
        --encoder openai-community/gpt2 --dim 768 --limit 4000 \
        --checkpoint-dir checkpoints/packs/public-gpt2 --device cuda
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from training.recipe.train_heads import _make_encoder


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RLCD calibration of noul heads")
    parser.add_argument("--specs", choices=["support", "public"], default="support")
    parser.add_argument("--encoder", default="hash", help="'hash' или model_id")
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--limit", type=int, default=0, help="размер датасета")
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None, help="куда сохранить головы")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--g", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def _load_splits(args):
    from eval.datasets.text_dataset import (
        QUESTIONS,
        generate_examples,
        spec_from_api,
        split_examples,
    )

    if args.specs == "public":
        from eval.datasets.public_dataset import PUBLIC_SPECS, load_ag_news

        base = load_ag_news(args.limit or None)
        carved = split_examples(
            base["train"], val_fraction=0.2, test_fraction=0.0, seed=args.seed
        )
        return PUBLIC_SPECS, carved["train"], carved["val"]

    specs = {name: spec_from_api(data) for name, data in QUESTIONS.items()}
    examples = generate_examples(n=args.limit or 1000, seed=args.seed)
    splits = split_examples(
        examples, val_fraction=0.2, test_fraction=0.2, seed=args.seed
    )
    return specs, splits["train"], splits["val"]


def main(argv: list[str] | None = None) -> dict:
    from eval.benchmark.core import _build_targets, _metrics_report, encode_examples
    from src.decision_head.heads import DecisionHeads
    from src.sampler.questions import questions_signature
    from training.rlcd.train import calibrate_noul_heads

    args = _parser().parse_args(argv)
    torch.manual_seed(args.seed)

    specs, train, val = _load_splits(args)
    encoder = _make_encoder(args.encoder, args.dim)

    train_features = encode_examples(specs, train, encoder)
    if args.checkpoint_dir is not None:
        from src.decision_head.heads import build_option_embeddings

        heads = DecisionHeads.load_checkpoint(
            Path(args.checkpoint_dir) / "heads.pt",
            seed=args.seed,
            option_embeddings=build_option_embeddings(specs, encoder.encode),
        )
    else:
        dim = next(iter(train_features.values())).shape[-1]
        heads = DecisionHeads(dim, specs, seed=args.seed)

    val_features = encode_examples(specs, val, encoder)
    train_targets = {
        name: _build_targets(spec, [ex["labels"][name] for ex in train])
        for name, spec in specs.items()
    }
    val_targets = {
        name: _build_targets(spec, [ex["labels"][name] for ex in val])
        for name, spec in specs.items()
    }

    before = _metrics_report(specs, val, val_features, heads)
    history = calibrate_noul_heads(
        heads,
        train_features,
        train_targets,
        steps=args.steps,
        lr=args.lr,
        g=args.g,
        gate_features=val_features,
        gate_targets=val_targets,
    )
    after = _metrics_report(specs, val, val_features, heads)

    print(f"RLCD-калибровка noul (specs={args.specs}, encoder={args.encoder})")
    print(f"  noul mean|p-o| (val) = {_mean(history):.4f}")
    for name in specs:
        b0, b1 = before["questions"][name], after["questions"][name]
        print(
            f"  {name:<12} brier {b0['brier']:.4f}->{b1['brier']:.4f}  "
            f"ece {b0['ece']:.4f}->{b1['ece']:.4f}  "
            f"acc {b0['accuracy']:.3f}->{b1['accuracy']:.3f}"
        )
    print(
        f"  overall acc {before['overall_accuracy']:.4f}->{after['overall_accuracy']:.4f}"
    )

    if args.out is not None:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        heads.save_checkpoint(out_dir / "heads.pt")
        heads.save_checkpoint(out_dir / f"{questions_signature(specs)}.pt")
        summary = {
            "specs": args.specs,
            "encoder": args.encoder,
            "feature_dim": heads.dim,
            "steps": args.steps,
            "lr": args.lr,
            "g": args.g,
            "seed": args.seed,
            "noul_mean_abs_error_val": history,
            "before": before,
            "after": after,
        }
        (out_dir / "calibration.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"  saved -> {out_dir}")

    return {"before": before, "after": after, "noul": history}


def _mean(history: dict[str, float]) -> float:
    if not history:
        return float("nan")
    return sum(history.values()) / len(history)


if __name__ == "__main__":
    main()
