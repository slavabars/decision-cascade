"""Измеряет точность внешней модели по вопросам на отложенном срезе домена.

Benefit-aware каскад (`src/backends/benefit.py`) эскалирует вопрос, только если
внешняя модель на нём точнее локальной с учётом цены вызова. Значит, ``q`` —
точность внешней модели по вопросу — надо измерить **до** эксплуатации и не на
том же test, где считается выигрыш.

Скрипт берёт train-срез домена (для RouterBench и AG News это их train-сплит,
для синтетических тикетов — тот же генератор с другим ``--fit-seed``), гоняет
по нему внешний SystemOne-сервис и пишет точность по вопросам:

    python -m eval.competitors.fit_remote_accuracy --specs public --limit 100 \
        --remote clef-flash=http://127.0.0.1:8091 \
        --out data/competitors_real/public_clef_accuracy.json

Файл потом передаётся каскаду как ``--remote-accuracy`` (в eval) или через
``DECISION_CASCADE_REMOTE_ACCURACY`` (в сервисе).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from random import Random
from typing import Any

from eval.competitors.evaluate import evaluate_backend
from eval.competitors.run import _parse_remotes, _specs
from src.backends.http import HttpSystemOneBackend


def fit_examples(
    kind: str, n: int, *, seed: int, shot: str
) -> list[dict[str, Any]]:
    """Отложенный (не test) срез домена для измерения точности внешней модели."""
    if kind == "support":
        from eval.datasets.text_dataset import generate_examples

        return generate_examples(n=n, seed=seed)
    if kind == "routerbench":
        from eval.datasets.routerbench import load_routerbench

        return load_routerbench(n or None, shot=shot)["train"][:n]
    from eval.datasets.public_dataset import load_ag_news

    # train AG News идёт по классам подряд: срез «первые n» — это один класс,
    # поэтому берём случайную подвыборку перемешанного пула.
    pool = load_ag_news(max(n * 20, 2000))["train"]
    return Random(seed).sample(pool, min(n, len(pool)))


def wilson_lower(accuracy: float, n: int, z: float = 1.96) -> float:
    """Нижняя граница 95% интервала Уилсона для доли правильных ответов.

    ``accuracy`` на срезе в 100 примеров — шумная оценка: 0.40 ± 0.1. Каскад,
    который верит точечной оценке, платит за эскалации там, где внешняя модель
    на самом деле не лучше локальной, поэтому в политике есть консервативный
    режим (``--remote-accuracy-bound lower``).
    """
    if n <= 0:
        return 0.0
    p = min(max(accuracy, 0.0), 1.0)
    denominator = 1.0 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return max(0.0, (centre - margin) / denominator)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Fit remote accuracy per question")
    parser.add_argument(
        "--specs", choices=["support", "public", "routerbench"], required=True
    )
    parser.add_argument("--limit", type=int, default=100, help="размер среза")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--fit-seed",
        type=int,
        default=10_000,
        help="seed генератора для support: не должен совпадать с замерным",
    )
    parser.add_argument("--shot", default="0shot")
    parser.add_argument(
        "--remote",
        action="append",
        default=[],
        metavar="NAME=URL",
        help="внешний SystemOne-совместимый сервис (ровно один)",
    )
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    names_urls = _parse_remotes(args.remote)
    if len(names_urls) != 1:
        raise SystemExit("нужен ровно один --remote NAME=URL")
    name, url = names_urls[0]

    specs = _specs(args.specs)
    examples = fit_examples(
        args.specs, args.limit, seed=args.fit_seed, shot=args.shot
    )
    if not examples:
        raise SystemExit("пустой срез: увеличьте --limit")

    fallback = HttpSystemOneBackend(url, model=name, timeout=args.timeout)
    try:
        report = evaluate_backend(
            specs, examples, fallback.decide, name=fallback.name, warmup=0
        )
    finally:
        close = getattr(fallback, "close", None)
        if callable(close):
            close()

    per_question = {
        name: metrics["accuracy"] for name, metrics in report["questions"].items()
    }
    n = report["n"]
    payload = {
        "specs": args.specs,
        "remote": fallback.name,
        "split": "train" if args.specs != "support" else f"seed={args.fit_seed}",
        "n": n,
        "accuracy": report["overall_accuracy"],
        "accuracy_lower": wilson_lower(report["overall_accuracy"], n),
        "per_question": per_question,
        "per_question_lower": {
            name: wilson_lower(accuracy, n) for name, accuracy in per_question.items()
        },
        "latency_ms": report["latency_ms"],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"{fallback.name} на {payload['split']} (n={payload['n']}): "
        f"accuracy {payload['accuracy']:.3f}"
    )
    for name, accuracy in payload["per_question"].items():
        print(f"  {name:<12} {accuracy:.3f}")
    print(f"report -> {args.out}")


if __name__ == "__main__":
    main()
