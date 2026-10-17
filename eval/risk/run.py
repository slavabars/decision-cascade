"""CLI: пороговый каскад против risk-aware на одном наборе задач.

Обучает головы, подгоняет оценщик риска на валидации и сравнивает на тесте:
локальные решения, каскад по порогу уверенности и каскад по бюджету риска.
Сохраняет ``risk.json`` (оценщик риска) и ``metrics.json`` (все числа).

Пример:
    python -m eval.risk.run --specs routerbench --encoder openai-community/gpt2 \\
        --dim 768 --train-steps 500 --limit 4000 --device cuda \\
        --out-dir data/risk_routerbench
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from eval.benchmark.core import encode_examples, train_benchmark
from eval.report.run import _encoder, _load_splits, _specs
from eval.risk.core import (
    GATE_RANKER_METRICS,
    GATE_TOL,
    cascade_summary,
    decisions_for_examples,
    flatten,
    gate_risk_vs_confidence,
    ranker_report,
    risk_escalation,
    selective_accuracy,
    threshold_escalation,
)
from training.recipe.risk import fit_error_risk_model

DEFAULT_THRESHOLDS = (0.5, 0.6, 0.7, 0.8)
DEFAULT_BUDGETS = (0.05, 0.15, 0.3, 0.5)


def _csv_floats(value: str) -> tuple[float, ...]:
    return tuple(float(part) for part in value.split(",") if part.strip())


def _print_table(title: str, rows: dict[str, dict[str, float]]) -> None:
    print(title)
    print(f"  {'setting':>8} {'escal%':>8} {'acc(acc)':>9} {'risk(acc)':>10} {'end2end':>8}")
    for key, metrics in rows.items():
        acc = metrics["accepted_accuracy"]
        risk = metrics["accepted_risk"]
        acc_str = f"{acc:9.3f}" if np.isfinite(acc) else f"{'n/a':>9}"
        risk_str = f"{risk:10.3f}" if np.isfinite(risk) else f"{'n/a':>10}"
        print(
            f"  {key:>8} {100 * metrics['escalation_rate']:>7.1f}% "
            f"{acc_str} {risk_str} {metrics['end2end_accuracy']:>8.3f}"
        )


def evaluate(
    specs,
    heads,
    encoder,
    val: list[dict[str, Any]],
    test: list[dict[str, Any]],
    *,
    seed: int = 0,
    thresholds=DEFAULT_THRESHOLDS,
    budgets=DEFAULT_BUDGETS,
) -> dict[str, Any]:
    """Подгоняет оценщик риска на ``val`` и сравнивает каскады на ``test``."""
    val_features = encode_examples(specs, val, encoder)
    val_decisions = decisions_for_examples(heads, val_features, len(val))
    val_flat, val_correct = flatten(specs, val_decisions, val)
    model = fit_error_risk_model(val_flat, val_correct, seed=seed)

    test_features = encode_examples(specs, test, encoder)
    test_decisions = decisions_for_examples(heads, test_features, len(test))
    test_flat, test_correct = flatten(specs, test_decisions, test)

    threshold_rows = {
        f"{t:.2f}": cascade_summary(test_correct, threshold_escalation(test_flat, t))
        for t in thresholds
    }
    risk_rows = {
        f"{b:.2f}": cascade_summary(test_correct, risk_escalation(test_flat, model, b))
        for b in budgets
    }
    confidence_scores = np.asarray([d.confidence for d in test_flat], dtype=float)
    risk_scores = np.asarray(
        [1.0 - model.predict_error(d) for d in test_flat], dtype=float
    )

    ranking = {
        "confidence": ranker_report(confidence_scores, test_correct),
        "risk_model": ranker_report(risk_scores, test_correct),
    }
    # hard gate: риск-ранжировщик принимается только если не хуже confidence
    gate_accepted, gate_worse = gate_risk_vs_confidence(
        ranking["confidence"], ranking["risk_model"], GATE_RANKER_METRICS
    )
    gate = {
        "accepted": gate_accepted,
        "worse": gate_worse,
        "baseline": "confidence",
        "candidate": "risk_model",
        "tol": GATE_TOL,
        "metrics": ranking,
    }

    return {
        "model": model,
        "local_accuracy": float(test_correct.mean()),
        "n_val": len(val),
        "n_test": len(test),
        "threshold_cascade": threshold_rows,
        "risk_cascade": risk_rows,
        "ranking": ranking,
        "gate": gate,
        "acceptance": {
            "confidence": selective_accuracy(confidence_scores, test_correct),
            "risk_model": selective_accuracy(risk_scores, test_correct),
        },
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Threshold vs risk-aware cascade")
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
    parser.add_argument(
        "--thresholds",
        type=_csv_floats,
        default=DEFAULT_THRESHOLDS,
        help="пороги уверенности порогового каскада, через запятую",
    )
    parser.add_argument(
        "--budgets",
        type=_csv_floats,
        default=DEFAULT_BUDGETS,
        help="бюджеты P(ошибка) risk-aware каскада, через запятую",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("data/risk"))
    args = parser.parse_args(argv)

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

    report = evaluate(
        specs,
        heads,
        encoder,
        val,
        test,
        seed=args.seed,
        thresholds=args.thresholds,
        budgets=args.budgets,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    # сертификат гейта уезжает в risk.json: риск-каскад в проде читает его и
    # при отказе фолбэчит на порог уверенности
    report["model"].gate = report["gate"]
    report["model"].save(args.out_dir / "risk.json")
    payload = {key: value for key, value in report.items() if key != "model"}
    payload.update(specs=args.specs, encoder=args.encoder, feature_dim=args.dim)
    (args.out_dir / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(f"local accuracy (test) = {report['local_accuracy']:.3f}")
    _print_table("threshold cascade (confidence < t):", report["threshold_cascade"])
    _print_table("risk-aware cascade (P(error) > budget):", report["risk_cascade"])
    print("ranking (lower AURC/eAURC is better):")
    for name, metrics in report["ranking"].items():
        print(
            f"  {name:>11}: AURC={metrics['aurc']:.4f} "
            f"eAURC={metrics['eaurc']:.4f} "
            f"cov@5%={metrics['coverage_at_5pct_risk']:.3f} "
            f"cov@10%={metrics['coverage_at_10pct_risk']:.3f}"
        )
    print("quality among accepted at equal escalation rates:")
    print(f"  {'accept%':>8} {'confidence':>11} {'risk_model':>11}")
    for coverage in report["acceptance"]["confidence"]:
        confidence_value = report["acceptance"]["confidence"][coverage]
        risk_value = report["acceptance"]["risk_model"][coverage]
        print(
            f"  {coverage:>8} {confidence_value:>11.3f} {risk_value:>11.3f}"
        )
    gate = report["gate"]
    verdict = "ПРИНЯТ" if gate["accepted"] else "ОТКАЗАН (фолбэк на confidence)"
    print(f"gate (risk vs confidence): {verdict} [tol={gate['tol']:g}]")
    for reason in gate["worse"]:
        print(f"  регрессия: {reason}")
    print(f"metrics -> {args.out_dir / 'metrics.json'}")
    print(f"risk    -> {args.out_dir / 'risk.json'}")


if __name__ == "__main__":
    main()
