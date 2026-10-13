"""CLI: сравнение наших бэкендов с внешними SystemOne-совместимыми сервисами.

Один набор размеченных примеров и один набор вопросов для всех участников.
Меряется качество (accuracy/Brier/ECE), латентность (p50/p95/p99), а для каскада
— доля эскалаций во внешний сервис (прямая экономия внешних вызовов).

Воспроизводимый прогон без сети (mock-конкурент):

    python -m eval.competitors.run --specs support --limit 200 \
        --mock --mock-latency-ms 5 --out data/competitors.json

Реальные конкуренты (Clef-Flash/vLLM/SGLang, Laya, Perplexity Decider):

    python -m eval.competitors.run --specs public --limit 200 \
        --remote clef-flash=http://127.0.0.1:8000 \
        --remote laya=https://api.laya.example \
        --out data/competitors.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from eval.competitors.evaluate import evaluate_backend
from eval.datasets.text_dataset import QUESTIONS as SUPPORT_QUESTIONS
from eval.datasets.text_dataset import generate_examples, spec_from_api
from src.backends.base import Backend
from src.backends.cascade import CascadeBackend
from src.backends.http import HttpSystemOneBackend
from src.backends.local import LocalHeadsBackend
from src.sampler.questions import QuestionSpec


def _specs(kind: str) -> dict[str, QuestionSpec]:
    if kind == "support":
        source = SUPPORT_QUESTIONS
    else:
        from eval.datasets.public_dataset import PUBLIC_QUESTIONS

        source = PUBLIC_QUESTIONS
    return {name: spec_from_api(data) for name, data in source.items()}


def _examples(kind: str, limit: int, seed: int) -> list[dict[str, Any]]:
    if kind == "support":
        return generate_examples(n=limit, seed=seed)
    from eval.datasets.public_dataset import load_ag_news

    return load_ag_news(limit or None)["test"]


def _local_backend(args) -> LocalHeadsBackend:
    from api.server.service import DecisionService
    from eval.report.run import _encoder

    encoder = None
    if args.encoder:
        encoder = _encoder(args.encoder, args.dim, args.device, args.dtype)
    service = DecisionService(
        encoder=encoder,
        dim=args.dim,
        checkpoint_dir=args.checkpoint_dir,
        seed=args.seed,
    )
    return LocalHeadsBackend(service)


def _parse_remotes(values: list[str]) -> list[tuple[str, str]]:
    remotes = []
    for value in values:
        if "=" not in value:
            raise SystemExit(f"--remote expects NAME=URL, got {value!r}")
        name, url = value.split("=", 1)
        remotes.append((name.strip(), url.strip()))
    return remotes


def _build_targets(args) -> list[Backend]:
    local = _local_backend(args)
    targets: list[Backend] = [local]

    fallback: Backend | None = None
    if args.mock:
        from eval.competitors.mocks import mock_systemone_transport

        mock = HttpSystemOneBackend(
            "http://mock",
            model="mock",
            transport=mock_systemone_transport(
                seed=args.seed, latency_ms=args.mock_latency_ms
            ),
        )
        targets.append(mock)
        fallback = mock

    for name, url in _parse_remotes(args.remote):
        remote = HttpSystemOneBackend(url, model=name, timeout=args.timeout)
        targets.append(remote)
        fallback = fallback or remote

    if fallback is not None:
        targets.append(CascadeBackend(local, fallback, threshold=args.threshold))
    return targets


def _print_table(reports: list[dict[str, Any]]) -> None:
    header = (
        f"{'target':<28} {'acc':>7} {'p50ms':>8} {'p95ms':>8} {'esc%':>7}"
    )
    print(header)
    print("-" * len(header))
    for report in reports:
        latency = report["latency_ms"]
        print(
            f"{report['name']:<28} {report['overall_accuracy']:>7.3f} "
            f"{latency.get('p50', float('nan')):>8.2f} "
            f"{latency.get('p95', float('nan')):>8.2f} "
            f"{100 * report['escalation_rate']:>7.2f}"
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Competitor benchmark")
    parser.add_argument("--specs", choices=["support", "public"], default="support")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--mock", action="store_true", help="добавить mock-конкурента")
    parser.add_argument("--mock-latency-ms", type=float, default=0.0)
    parser.add_argument(
        "--remote",
        action="append",
        default=[],
        metavar="NAME=URL",
        help="внешний SystemOne-совместимый сервис (можно несколько раз)",
    )
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--encoder", default="", help="'' (hash) или model_id")
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--out", type=Path, default=Path("data/competitors.json"))
    args = parser.parse_args(argv)

    specs = _specs(args.specs)
    examples = _examples(args.specs, args.limit, args.seed)
    print(
        f"specs={args.specs} n={len(examples)} questions={len(specs)} "
        f"threshold={args.threshold}"
    )

    targets = _build_targets(args)
    reports = []
    try:
        for target in targets:
            report = evaluate_backend(
                specs, examples, target.decide, name=target.name, warmup=args.warmup
            )
            reports.append(report)
    finally:
        for target in targets:
            close = getattr(target, "close", None)
            if callable(close):
                close()

    _print_table(reports)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "specs": args.specs,
        "n": len(examples),
        "threshold": args.threshold,
        "reports": reports,
    }
    args.out.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"report -> {args.out}")


if __name__ == "__main__":
    main()
