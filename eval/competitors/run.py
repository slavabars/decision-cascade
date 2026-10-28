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

Публичный decision-датасет RouterBench (маршрутизация LLM) и кривая
«качество против доли эскалаций» (``--sweep``):

    python -m eval.competitors.run --specs routerbench --limit 500 \
        --mock --mock-latency-ms 2 --sweep --out data/competitors_routerbench.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from eval.competitors.evaluate import evaluate_backend
from eval.datasets.routerbench import ROUTERBENCH_FILES
from eval.datasets.text_dataset import QUESTIONS as SUPPORT_QUESTIONS
from eval.datasets.text_dataset import generate_examples, spec_from_api
from src.backends.base import Backend
from src.backends.benefit import BenefitAwareCascadeBackend, load_remote_accuracy
from src.backends.cascade import CascadeBackend
from src.backends.http import HttpSystemOneBackend
from src.backends.local import LocalHeadsBackend
from src.sampler.questions import QuestionSpec


def _specs(kind: str) -> dict[str, QuestionSpec]:
    if kind == "support":
        source = SUPPORT_QUESTIONS
    elif kind == "routerbench":
        from eval.datasets.routerbench import ROUTERBENCH_QUESTIONS

        source = ROUTERBENCH_QUESTIONS
    else:
        from eval.datasets.public_dataset import PUBLIC_QUESTIONS

        source = PUBLIC_QUESTIONS
    return {name: spec_from_api(data) for name, data in source.items()}


def _examples(
    kind: str, limit: int, seed: int, shot: str = "0shot"
) -> list[dict[str, Any]]:
    if kind == "support":
        return generate_examples(n=limit, seed=seed)
    if kind == "routerbench":
        from eval.datasets.routerbench import load_routerbench

        return load_routerbench(limit or None, shot=shot)["test"]
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


def _build_fallbacks(args) -> list[Backend]:
    """Внешние бэкенды-участники; первый становится fallback каскада."""
    fallbacks: list[Backend] = []
    if args.mock:
        from eval.competitors.mocks import mock_systemone_transport

        fallbacks.append(
            HttpSystemOneBackend(
                "http://mock",
                model="mock",
                transport=mock_systemone_transport(
                    seed=args.seed, latency_ms=args.mock_latency_ms
                ),
            )
        )
    for name, url in _parse_remotes(args.remote):
        fallbacks.append(HttpSystemOneBackend(url, model=name, timeout=args.timeout))
    return fallbacks


def _threshold_grid(points: int) -> list[float]:
    points = max(2, points)
    return [i / (points - 1) for i in range(points)]


def _remote_accuracy(args) -> float | dict[str, float]:
    """Кэшированный ``q`` для benefit-политики (число или JSON от fit-скрипта)."""
    if getattr(args, "_remote_accuracy", None) is None:
        if not args.remote_accuracy:
            raise SystemExit(
                "--policy benefit requires --remote-accuracy "
                "(число или JSON из eval.competitors.fit_remote_accuracy)"
            )
        args._remote_accuracy = load_remote_accuracy(
            args.remote_accuracy,
            conservative=args.remote_accuracy_bound == "lower",
        )
    return args._remote_accuracy


def _build_cascade(
    args, local: Backend, fallback: Backend, parameter: float | None = None
) -> CascadeBackend:
    """Каскад заданной политики; ``parameter`` — порог или цена эскалации."""
    if args.policy == "benefit":
        return BenefitAwareCascadeBackend(
            local,
            fallback,
            _remote_accuracy(args),
            escalation_cost=(
                args.escalation_cost if parameter is None else parameter
            ),
            error_cost=args.error_cost,
        )
    return CascadeBackend(
        local,
        fallback,
        threshold=args.threshold if parameter is None else parameter,
    )


def _sweep_cascade(
    args,
    local: Backend,
    fallback: Backend,
    specs: dict[str, QuestionSpec],
    examples: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Кривая «качество против доли эскалаций».

    Для ``--policy confidence`` параметр — порог уверенности, для
    ``--policy benefit`` — цена эскалации (в единицах «верный ответ = 1»).
    """
    rows: list[dict[str, Any]] = []
    for parameter in _threshold_grid(args.sweep_points):
        cascade = _build_cascade(args, local, fallback, parameter)
        report = evaluate_backend(
            specs, examples, cascade.decide, name=cascade.name, warmup=0
        )
        rows.append(
            {
                "threshold": parameter,
                "overall_accuracy": report["overall_accuracy"],
                "escalation_rate": report["escalation_rate"],
                "requests_with_escalation": report["requests_with_escalation"],
                "latency_ms": report["latency_ms"],
            }
        )
    return rows


def _print_table(reports: list[dict[str, Any]]) -> None:
    header = f"{'target':<28} {'acc':>7} {'p50ms':>8} {'p95ms':>8} {'esc%':>7}"
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


def _print_sweep(fallback_name: str, rows: list[dict[str, Any]], policy: str) -> None:
    parameter = "cost" if policy == "benefit" else "threshold"
    print(f"cascade sweep -> {fallback_name} ({policy})")
    print(f"{parameter:>9} {'acc':>7} {'esc%':>7} {'p50ms':>8}")
    for row in rows:
        print(
            f"{row['threshold']:>9.2f} {row['overall_accuracy']:>7.3f} "
            f"{100 * row['escalation_rate']:>7.2f} "
            f"{row['latency_ms'].get('p50', float('nan')):>8.2f}"
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Competitor benchmark")
    parser.add_argument(
        "--specs", choices=["support", "public", "routerbench"], default="support"
    )
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument(
        "--policy",
        choices=["confidence", "benefit"],
        default="confidence",
        help=(
            "правило эскалации: порог уверенности или ожидаемая выгода "
            "(нужен --remote-accuracy)"
        ),
    )
    parser.add_argument(
        "--remote-accuracy",
        default="",
        help=(
            "точность внешней модели q: число или JSON из "
            "eval.competitors.fit_remote_accuracy (per_question)"
        ),
    )
    parser.add_argument(
        "--remote-accuracy-bound",
        choices=["point", "lower"],
        default="point",
        help=(
            "точечная q или её нижняя граница Уилсона (консервативно: "
            "меньше вредных эскалаций на шумной оценке)"
        ),
    )
    parser.add_argument(
        "--escalation-cost",
        type=float,
        default=0.05,
        help="цена эскалации C в единицах «верный ответ = 1»",
    )
    parser.add_argument(
        "--error-cost",
        type=float,
        default=1.0,
        help="цена ошибки D относительно верного ответа",
    )
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
    parser.add_argument(
        "--shot",
        choices=sorted(ROUTERBENCH_FILES),
        default="0shot",
        help="RouterBench: 0-shot или 5-shot",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="кривая качество/эскалация каскада при пороге от 0.0 до 1.0",
    )
    parser.add_argument("--sweep-points", type=int, default=11)
    parser.add_argument("--out", type=Path, default=Path("data/competitors.json"))
    args = parser.parse_args(argv)

    specs = _specs(args.specs)
    examples = _examples(args.specs, args.limit, args.seed, args.shot)
    print(
        f"specs={args.specs} n={len(examples)} questions={len(specs)} "
        f"policy={args.policy} "
        + (
            f"escalation_cost={args.escalation_cost} "
            if args.policy == "benefit"
            else f"threshold={args.threshold} "
        )
    )

    local = _local_backend(args)
    fallbacks = _build_fallbacks(args)
    targets: list[Backend] = [local, *fallbacks]
    primary_fallback = fallbacks[0] if fallbacks else None
    if primary_fallback is not None and not args.sweep:
        targets.append(_build_cascade(args, local, primary_fallback))
    if args.policy == "benefit":
        # q нужен и для отчёта, и для sweep — падаем сразу, а не после прогона
        _remote_accuracy(args)

    reports = []
    sweep_rows: list[dict[str, Any]] = []
    try:
        for target in targets:
            report = evaluate_backend(
                specs, examples, target.decide, name=target.name, warmup=args.warmup
            )
            reports.append(report)
        if args.sweep and primary_fallback is not None:
            sweep_rows = _sweep_cascade(
                args, local, primary_fallback, specs, examples
            )
    finally:
        for target in [local, *fallbacks]:
            close = getattr(target, "close", None)
            if callable(close):
                close()

    _print_table(reports)
    if sweep_rows:
        _print_sweep(primary_fallback.name, sweep_rows, args.policy)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "specs": args.specs,
        "n": len(examples),
        "policy": args.policy,
        "threshold": args.threshold,
        "reports": reports,
    }
    if args.policy == "benefit":
        payload["escalation_cost"] = args.escalation_cost
        payload["error_cost"] = args.error_cost
        payload["remote_accuracy"] = _remote_accuracy(args)
    if sweep_rows:
        payload["cascade_sweep"] = {
            "fallback": primary_fallback.name,
            "rows": sweep_rows,
        }
    args.out.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"report -> {args.out}")


if __name__ == "__main__":
    main()
