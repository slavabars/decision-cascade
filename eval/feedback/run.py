"""CLI: смещение ``q`` при фидбэке только по эскалациям и эффект IPS.

Пример:
    python -m eval.feedback.run --rounds 30 --rows 1000 --explore 0 0.05 \\
        --json data/feedback_sim/report.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval.feedback.core import SCENARIOS, report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bias of q estimated from escalations")
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=sorted(SCENARIOS),
        default=sorted(SCENARIOS),
        help="сценарии корреляции трудности и качества внешней модели",
    )
    parser.add_argument(
        "--estimators",
        nargs="+",
        choices=["naive", "ips"],
        default=["naive", "ips"],
    )
    parser.add_argument(
        "--explore",
        type=float,
        nargs="+",
        default=[0.0, 0.05],
        help="доли исследовательских эскалаций",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--rows", type=int, default=1000)
    parser.add_argument("--prior", type=float, default=0.75)
    parser.add_argument("--shrinkage", type=float, default=50.0)
    parser.add_argument("--cost", type=float, default=0.05)
    parser.add_argument("--error-cost", type=float, default=1.0)
    parser.add_argument("--json", type=Path, default=None, help="куда сохранить отчёт")
    return parser


def _print(rows: list[dict]) -> None:
    current = None
    for row in rows:
        if row["scenario"] != current:
            current = row["scenario"]
            print(
                f"\nscenario={row['scenario']}  q_true={row['q_true']:.4f}  "
                f"prior={row['prior']:.2f}  W={row['shrinkage']:.0f}  "
                f"({row['description']})"
            )
            print(
                f"{'estimator':>9} {'explore':>7} {'q_hat':>14} {'bias':>16} "
                f"{'policy_rate':>11} {'calls_rate':>10} {'regret':>8}"
            )
        print(
            f"{row['estimator']:>9} {row['explore']:>7.2f} "
            f"{row['q_hat_mean']:>7.4f}±{row['q_hat_std']:.4f} "
            f"{row['bias_mean']:>+9.4f}±{row['bias_std']:.4f} "
            f"{row['policy_rate']:>11.3f} {row['calls_rate']:>10.3f} "
            f"{row['regret_rate']:>8.4f}"
        )


def main(argv: list[str] | None = None) -> list[dict]:
    args = _parser().parse_args(argv)
    rows = report(
        scenarios=args.scenarios,
        estimators=args.estimators,
        explores=args.explore,
        seeds=args.seeds,
        rounds=args.rounds,
        rows=args.rows,
        prior=args.prior,
        shrinkage=args.shrinkage,
        cost=args.cost,
        error_cost=args.error_cost,
    )
    _print(rows)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\nreport -> {args.json}")
    return rows


if __name__ == "__main__":  # pragma: no cover
    main()
