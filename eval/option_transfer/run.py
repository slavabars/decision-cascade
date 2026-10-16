"""CLI: zero-shot перенос вариантов выбора (linear vs option_scoring).

Голова выбора обучается только на трёх вариантах, а на инференсе ей подменяют
набор вариантов теми же весами. Сравнивается с головой, обученной на этом наборе
с нуля (верхняя граница). Метрики усредняются по нескольким seed'ам.

Пример (офлайн, hash-энкодер):
    python -m eval.option_transfer.run --encoder hash --dim 64

Семантический перенос требует реального энкодера:
    python -m eval.option_transfer.run --encoder openai-community/gpt2 --dim 768
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval.option_transfer.core import run_option_transfer
from eval.report.run import _encoder


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Option-set transfer for choice heads")
    parser.add_argument("--encoder", default="hash")
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--train-steps", type=int, default=800)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--n-per-label", type=int, default=40)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--out-dir", type=Path, default=Path("data/option_transfer"))
    args = parser.parse_args(argv)

    encoder = _encoder(args.encoder, args.dim, args.device, args.dtype)
    report = run_option_transfer(
        encoder,
        train_steps=args.train_steps,
        lr=args.lr,
        seeds=tuple(args.seeds),
        n_per_label=args.n_per_label,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    _print(report)
    print(f"metrics -> {args.out_dir / 'metrics.json'}")


def _cell(summary: dict, key: str) -> str:
    if key not in summary:
        return "n/a"
    entry = summary[key]
    return f"{entry['mean']:.3f}±{entry['std']:.3f}"


def _print(report: dict) -> None:
    summary = report["summary"]
    print(f"encoder: {report['encoder']}   seeds: {report['seeds']}")
    print(f"seen options:   {report['seen_options']}")
    print(f"unseen options: {report['unseen_options']}")
    print(f"new options:    {report['new_options']}")
    print()

    rows = [
        ("seen (in-dist)", "option_scoring", "seen.option_scoring",
         f"chance {1 / len(report['seen_options']):.3f}"),
        ("seen (in-dist)", "linear", "seen.linear",
         f"chance {1 / len(report['seen_options']):.3f}"),
        ("zero-shot 2way", "option_scoring", "zero_shot.option_scoring",
         f"chance {1 / 2:.3f}"),
        ("zero-shot 2way", "linear-retrained", "zero_shot.linear_retrained",
         "upper bound"),
        ("with distractors", "option_scoring", "distractors.option_scoring",
         f"chance {1 / 5:.3f}"),
        ("new words", "option_scoring", "new_words.option_scoring",
         f"chance {1 / 2:.3f}"),
        ("new words", "linear-retrained", "new_words.linear_retrained",
         "upper bound"),
    ]
    print(f"{'setting':>18} {'head':>18} {'accuracy':>16}  note")
    for setting, head, key, note in rows:
        print(f"{setting:>18} {head:>18} {_cell(summary, key):>16}  {note}")

    width = summary.get("linear_option_set_width", {}).get("mean")
    print()
    print(
        "linear output width is fixed at training time:"
        f" {int(width) if width else '?'} options — it cannot serve a different set"
    )


if __name__ == "__main__":
    main()
