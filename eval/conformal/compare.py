"""CLI: сравнить валидные методы выбора conformal-порога на одних данных.

Один раз обучает головы и риск-модель, делит val на модель риска и калибровку,
а затем строит сертификаты тремя методами и проверяет их на отложенном тесте:

- ``fixed_sequence`` — продуктовый метод (fixed-sequence по сетке);
- ``bonferroni_grid`` — Бонферрони по фиксированной сетке;
- ``split_half`` — выбор на одной половине, сертификация на другой.

Пример:
    python -m eval.conformal.compare --specs public \\
        --encoder openai-community/gpt2 --dim 768 --train-steps 300 \\
        --limit 32000 --alpha 0.05 0.1 --device cuda \\
        --json data/conformal_compare/report.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from eval.benchmark.core import train_benchmark
from eval.conformal.core import (
    certificate_from_scores,
    evaluate_certificate,
    prepare_evaluation,
)
from eval.conformal.methods import bonferroni_grid, split_half
from eval.report.run import _encoder, _load_splits, _specs


def _evaluate(
    certificate: dict[str, Any], data: dict[str, Any]
) -> dict[str, Any]:
    return evaluate_certificate(
        certificate,
        data["test_scores"],
        data["test_correct"],
        data["confidence_scores"],
        groups=data["test_groups"],
    )


def compare(
    data: dict[str, Any],
    *,
    alphas: list[float],
    delta: float,
    min_coverages: list[float],
    grid_points: list[int],
    split_seed: int,
    split_repeats: int,
) -> list[dict[str, Any]]:
    """Сетка alpha × min_coverage × метод; строки готовы к печати и JSON."""
    rows: list[dict[str, Any]] = []
    for alpha in alphas:
        for min_coverage in min_coverages:
            fixed = certificate_from_scores(
                data["calibration_scores"],
                data["calibration_correct"],
                data["calibration_groups"],
                alpha=alpha,
                delta=delta,
                min_coverage=min_coverage,
            )
            gate = _evaluate(fixed, data)
            rows.append(
                {
                    "alpha": alpha,
                    "min_coverage": min_coverage,
                    "method": "fixed_sequence",
                    "calibration_coverage": fixed["coverage"],
                    "calibration_risk": fixed["empirical_risk"],
                    "calibration_upper": fixed["upper_bound"],
                    "test_coverage": gate["test"]["coverage"],
                    "test_risk": gate["test"]["risk"],
                    "gate": gate["accepted"],
                }
            )

            for points in grid_points:
                cert = bonferroni_grid(
                    data["calibration_scores"],
                    data["calibration_correct"],
                    data["calibration_groups"],
                    alpha=alpha,
                    delta=delta,
                    min_coverage=min_coverage,
                    grid_points=points,
                )
                gate = _evaluate(cert, data)
                rows.append(
                    {
                        "alpha": alpha,
                        "min_coverage": min_coverage,
                        "method": f"bonferroni_grid_{points}",
                        "calibration_coverage": cert["coverage"],
                        "calibration_risk": cert["empirical_risk"],
                        "calibration_upper": cert["upper_bound"],
                        "test_coverage": gate["test"]["coverage"],
                        "test_risk": gate["test"]["risk"],
                        "gate": gate["accepted"],
                    }
                )

            split = split_half(
                data["calibration_scores"],
                data["calibration_correct"],
                data["calibration_groups"],
                alpha=alpha,
                delta=delta,
                min_coverage=min_coverage,
                seed=split_seed,
                repeats=split_repeats,
            )
            test_coverages = []
            test_risks = []
            for run in split["runs"]:
                if not run["passed"]:
                    continue
                taken = data["test_scores"] <= run["threshold"]
                if not taken.any():
                    continue
                test_coverages.append(float(taken.mean()))
                test_risks.append(float(1.0 - data["test_correct"][taken].mean()))
            rows.append(
                {
                    "alpha": alpha,
                    "min_coverage": min_coverage,
                    "method": "split_half",
                    "calibration_coverage": split["coverage_among_pass"],
                    "calibration_risk": None,
                    "calibration_upper": None,
                    "test_coverage": (
                        float(np.mean(test_coverages)) if test_coverages else 0.0
                    ),
                    "test_risk": float(np.mean(test_risks)) if test_risks else None,
                    "gate": split["pass_rate"] > 0,
                    "pass_rate": split["pass_rate"],
                    "coverage_mean": split["coverage_mean"],
                }
            )
    return rows


def _print(rows: list[dict[str, Any]], data: dict[str, Any]) -> None:
    print(
        f"calibration examples={data['n_calibration']} "
        f"decisions={data['n_calibration_decisions']}, "
        f"test examples={data['n_test']} decisions={data['n_test_decisions']}, "
        f"local_accuracy={data['local_accuracy']:.3f}"
    )
    print(
        f"{'alpha':>5} {'min_cov':>7} {'method':>22} {'cal_cov':>8} "
        f"{'cal_risk':>8} {'cal_ub':>7} {'test_cov':>8} {'test_risk':>9} {'ok':>5}"
    )

    def fmt(value: Any, pattern: str) -> str:
        return "—" if value is None else pattern.format(value)

    for row in rows:
        ok = (
            f"{row['pass_rate']:.2f}"
            if "pass_rate" in row
            else str(row["gate"])
        )
        print(
            f"{row['alpha']:>5} {row['min_coverage']:>7} {row['method']:>22} "
            f"{fmt(row['calibration_coverage'], '{:.3f}'):>8} "
            f"{fmt(row['calibration_risk'], '{:.4f}'):>8} "
            f"{fmt(row['calibration_upper'], '{:.4f}'):>7} "
            f"{fmt(row['test_coverage'], '{:.3f}'):>8} "
            f"{fmt(row['test_risk'], '{:.4f}'):>9} {ok:>5}"
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare conformal threshold methods")
    parser.add_argument(
        "--specs", choices=["support", "public", "routerbench"], default="routerbench"
    )
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--encoder", default="hash")
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--train-steps", type=int, default=300)
    parser.add_argument("--rlcd-steps", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shot", choices=["0shot", "5shot"], default="0shot")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--alpha", type=float, nargs="+", default=[0.05, 0.1])
    parser.add_argument("--delta", type=float, default=0.1)
    parser.add_argument("--min-coverage", type=float, nargs="+", default=[0.0, 0.3, 0.5])
    parser.add_argument("--grid-points", type=int, nargs="+", default=[20, 50])
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--split-repeats", type=int, default=20)
    parser.add_argument("--json", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> list[dict[str, Any]]:
    args = _parser().parse_args(argv)
    specs = _specs(args.specs)
    splits = _load_splits(args)
    train, test = splits["train"], splits["test"]
    val = splits.get("val") or train
    print(
        f"specs={args.specs} encoder={args.encoder} "
        f"train={len(train)} val={len(val)} test={len(test)}"
    )

    encoder = _encoder(args.encoder, args.dim, args.device, args.dtype)
    heads, _ = train_benchmark(
        specs,
        train,
        encoder,
        train_steps=args.train_steps,
        lr=args.lr,
        seed=args.seed,
        rlcd_steps=args.rlcd_steps,
    )
    data = prepare_evaluation(specs, heads, encoder, val, test, seed=args.seed)
    rows = compare(
        data,
        alphas=args.alpha,
        delta=args.delta,
        min_coverages=args.min_coverage,
        grid_points=args.grid_points,
        split_seed=args.split_seed,
        split_repeats=args.split_repeats,
    )
    _print(rows, data)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(
                {
                    "specs": args.specs,
                    "encoder": args.encoder,
                    "feature_dim": args.dim,
                    "limit": args.limit,
                    "delta": args.delta,
                    "n_calibration": data["n_calibration"],
                    "n_calibration_decisions": data["n_calibration_decisions"],
                    "n_test": data["n_test"],
                    "n_test_decisions": data["n_test_decisions"],
                    "local_accuracy": data["local_accuracy"],
                    "rows": rows,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"report -> {args.json}")
    return rows


if __name__ == "__main__":  # pragma: no cover
    main()
