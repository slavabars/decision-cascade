"""CLI: conformal-сертификат selective prediction на отложенных данных.

Делит val на две части: на первой обучается оценщик ``P(ошибка)``, на второй
строится сертификат (порог с гарантией доли ошибок), тест остаётся для гейта
«не навреди». Сохраняет ``conformal.json`` (его читает прод) и ``metrics.json``.

Пример:
    python -m eval.conformal.run --specs public --encoder openai-community/gpt2 \\
        --dim 768 --train-steps 300 --limit 16000 --alpha 0.1 --delta 0.1 \\
        --min-coverage 0.5 --out-dir data/conformal_public
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from eval.benchmark.core import train_benchmark
from eval.conformal.core import (
    certificate_from_scores,
    evaluate_certificate,
    prepare_evaluation,
)
from eval.report.run import _encoder, _load_splits, _specs
from src.conformal import save_certificate


def evaluate(
    specs,
    heads,
    encoder,
    val: list[dict[str, Any]],
    test: list[dict[str, Any]],
    *,
    alpha: float,
    delta: float,
    min_coverage: float,
    seed: int = 0,
) -> dict[str, Any]:
    """Обучает риск-модель на половине val и строит сертификат на другой."""
    data = prepare_evaluation(specs, heads, encoder, val, test, seed=seed)
    certificate = certificate_from_scores(
        data["calibration_scores"],
        data["calibration_correct"],
        data["calibration_groups"],
        alpha=alpha,
        delta=delta,
        min_coverage=min_coverage,
    )
    gate = evaluate_certificate(
        certificate,
        data["test_scores"],
        data["test_correct"],
        data["confidence_scores"],
        groups=data["test_groups"],
    )
    certificate["gate"] = gate

    return {
        "model": data["model"],
        "certificate": certificate,
        "gate": gate,
        "local_accuracy": data["local_accuracy"],
        "n_train": None,
        "n_val_model": data["n_val_model"],
        "n_calibration": data["n_calibration"],
        "n_test": data["n_test"],
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Conformal selective certificate")
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
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--delta", type=float, default=0.1)
    parser.add_argument("--min-coverage", type=float, default=0.3)
    parser.add_argument("--out-dir", type=Path, default=Path("data/conformal"))
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
        alpha=args.alpha,
        delta=args.delta,
        min_coverage=args.min_coverage,
        seed=args.seed,
    )
    report["n_train"] = len(train)
    certificate = report["certificate"]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    save_certificate(certificate, args.out_dir / "conformal.json")
    payload = {
        key: value for key, value in report.items() if key != "model"
    }
    payload.update(
        specs=args.specs,
        encoder=args.encoder,
        feature_dim=args.dim,
        alpha=args.alpha,
        delta=args.delta,
        min_coverage=args.min_coverage,
    )
    (args.out_dir / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(f"local accuracy (test) = {report['local_accuracy']:.3f}")
    print(
        f"calibration n={certificate['n']} alpha={certificate['alpha']:.3f} "
        f"delta={certificate['delta']:.3f}"
    )
    if certificate["empirical_risk"] is None:
        print("certificate: покрытие 0 — данных не хватило даже на минимальное")
    else:
        print(
            f"certificate: threshold={certificate['threshold']:.4f} "
            f"coverage={certificate['coverage']:.3f} "
            f"risk={certificate['empirical_risk']:.4f} "
            f"upper={certificate['upper_bound']:.4f}"
        )
    gate = report["gate"]
    test_stats = gate["test"]
    print(
        f"test: coverage={test_stats['coverage']:.3f} "
        f"risk={test_stats['risk']:.4f} "
        f"confidence@coverage={test_stats['baseline_risk']:.4f}"
    )
    group_risks = [
        row["risk"]
        for row in test_stats.get("groups", {}).values()
        if row["risk"] is not None
    ]
    if group_risks:
        print(
            f"test groups: {len(group_risks)} "
            f"worst risk={max(group_risks):.4f}"
        )
    verdict = "ПРИНЯТ" if gate["accepted"] else "ОТКАЗАН (фолбэк на confidence)"
    print(f"gate (conformal vs confidence): {verdict}")
    for reason in gate["worse"]:
        print(f"  регрессия: {reason}")
    print(f"certificate -> {args.out_dir / 'conformal.json'}")
    print(f"metrics     -> {args.out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
