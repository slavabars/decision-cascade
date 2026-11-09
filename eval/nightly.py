"""Ночной регресс качества: опубликованные паки на фиксированных сплитах.

Сравнивает accuracy пака из релиза с базовой линией
(``eval/nightly_baseline.json``) и завершается с кодом 1, если падение больше
допуска (по умолчанию 1 п.п.). Сплиты детерминированы: AG News закреплён
ревизией, RouterBench — ревизией и сидом.

Пример:
    python -m eval.nightly --suite ag_news --device cuda
    python -m eval.nightly --suite all --packs-root checkpoints/packs \\
        --out data/nightly/report.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from api.server.service import DecisionService
from eval.competitors.evaluate import evaluate_backend
from eval.datasets.text_dataset import spec_from_api
from src.backends.local import LocalHeadsBackend
from src.device import resolve_device
from src.encoder.factory import create_state_encoder

SUITES = ("ag_news", "routerbench")
DEFAULT_BASELINE = Path("eval/nightly_baseline.json")
DEFAULT_PACKS = Path("checkpoints/packs")
#: 4000 -> train 3200 / val 800 / test 1000, ровно сплиты опубликованных паков.
DEFAULT_LIMIT = 4000


def load_pack(pack_dir: Path, device: str) -> tuple[LocalHeadsBackend, dict, dict]:
    """Собирает локальный бэкенд из пака (``metadata.json`` + ``heads.pt``)."""
    metadata = json.loads((pack_dir / "metadata.json").read_text(encoding="utf-8"))
    dim = int(metadata["feature_dim"])
    encoder = create_state_encoder(
        metadata["encoder"],
        dim,
        device=device,
        max_length=metadata.get("max_length", 256),
        batch_size=int(metadata.get("encode_batch", 16)),
    )
    encoder.load()
    specs = {
        name: spec_from_api(data) for name, data in metadata["questions"].items()
    }
    service = DecisionService(encoder=encoder, dim=dim, checkpoint_dir=pack_dir)
    loaded = service.warmup()
    if loaded == 0:
        raise RuntimeError(
            f"pack {pack_dir} is not readable: no checkpoint matched questions "
            f"{sorted(specs)}"
        )
    return LocalHeadsBackend(service, name=pack_dir.name), specs, metadata


def load_suite(suite: str, limit: int, seed: int) -> list[dict[str, Any]]:
    """Test-сплит набора: тот же срез, на котором измерялся пак."""
    if suite == "ag_news":
        from eval.datasets.public_dataset import load_ag_news

        return load_ag_news(limit)["test"]
    from eval.datasets.routerbench import load_routerbench

    return load_routerbench(limit, seed=seed)["test"]


def within_tolerance(accuracy: float, expected: float, allowed: float) -> bool:
    """Не считается регрессом падение в пределах допуска (включительно).

    Эпсилон гасит двоичную погрешность на границе: 0.87 против 0.88 при
    допуске 0.01 — это ровно 1 п.п., а не «чуть больше».
    """
    return accuracy - expected >= -allowed - 1e-9


def check_latency(
    entry: dict[str, Any], latency_ms: dict[str, float]
) -> dict[str, Any] | None:
    """Сверяет измеренные p50/p95 с бюджетом набора (``entry["latency"]``).

    ``None`` — бюджета нет или метрик нет: это не ошибка, просто нечего
    проверять. Допуск задаётся множителем, потому что CPU-раннеры дают разброс
    в десятки процентов на одинаковом коде.
    """
    budget = entry.get("latency")
    if not budget or not latency_ms:
        return None
    factor = float(budget.get("tolerance", 1.25))
    checks: dict[str, dict[str, Any]] = {}
    for key, metric in (("p50_ms", "p50"), ("p95_ms", "p95")):
        if key not in budget or metric not in latency_ms:
            continue
        limit = float(budget[key]) * factor
        value = float(latency_ms[metric])
        checks[key] = {
            "value_ms": value,
            "limit_ms": limit,
            "passed": value <= limit,
        }
    if not checks:
        return None
    return {
        "checks": checks,
        "passed": all(item["passed"] for item in checks.values()),
    }


def check_suite(
    suite: str,
    entry: dict[str, Any],
    packs_root: Path,
    *,
    device: str,
    limit: int,
    seed: int,
    tolerance: float | None,
) -> dict[str, Any]:
    """Меряет один пак и сравнивает с базовой линией."""
    pack_dir = packs_root / entry["pack"]
    backend, specs, _metadata = load_pack(pack_dir, device)
    examples = load_suite(suite, limit, seed)[:entry.get("n", limit)]
    metrics = evaluate_backend(specs, examples, backend.decide, name=backend.name)
    expected = float(entry["overall_accuracy"])
    allowed = float(entry.get("tolerance", 0.01) if tolerance is None else tolerance)
    delta = metrics["overall_accuracy"] - expected
    passed = within_tolerance(metrics["overall_accuracy"], expected, allowed)
    latency = check_latency(entry, metrics.get("latency_ms", {}))
    return {
        "suite": suite,
        "pack": entry["pack"],
        "n": metrics["n"],
        "accuracy": metrics["overall_accuracy"],
        "baseline": expected,
        "delta": delta,
        "tolerance": allowed,
        "passed": passed,
        "questions": metrics["questions"],
        "latency_ms": metrics.get("latency_ms", {}),
        "latency": latency,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Nightly quality regression")
    parser.add_argument(
        "--suite", choices=[*SUITES, "all"], default="all", help="набор или all"
    )
    parser.add_argument("--packs-root", type=Path, default=DEFAULT_PACKS)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--tolerance",
        type=float,
        default=None,
        help="допуск падения accuracy (по умолчанию из baseline, 0.01)",
    )
    parser.add_argument(
        "--latency",
        choices=["off", "warn", "fail"],
        default="warn",
        help="проверка бюджета латентности: off — не считать, warn — печатать, "
        "fail — ронять job (в ночном workflow)",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    selected = SUITES if args.suite == "all" else (args.suite,)
    missing = [suite for suite in selected if suite not in baseline]
    if missing:
        parser.error(f"baseline has no entry for {missing}")

    device = resolve_device(args.device)
    results: list[dict[str, Any]] = []
    for suite in selected:
        result = check_suite(
            suite,
            baseline[suite],
            args.packs_root,
            device=device,
            limit=args.limit,
            seed=args.seed,
            tolerance=args.tolerance,
        )
        results.append(result)
        status = "PASS" if result["passed"] else "REGRESSION"
        print(
            f"{status:<10} suite={suite:<12} pack={result['pack']:<20} "
            f"n={result['n']:<5} accuracy={result['accuracy']:.4f} "
            f"baseline={result['baseline']:.4f} delta={result['delta']:+.4f} "
            f"tolerance={result['tolerance']:.3f}",
            flush=True,
        )
        latency = result.get("latency")
        if latency is not None:
            parts = [
                f"{key}={check['value_ms']:.0f}/{check['limit_ms']:.0f}ms"
                for key, check in latency["checks"].items()
            ]
            print(
                f"{'':<10} latency {' '.join(parts)} "
                f"{'OK' if latency['passed'] else 'SLOW'}",
                flush=True,
            )

    checked = [result for result in results if result.get("latency")]
    latency_passed: bool | None
    if args.latency == "off" or not checked:
        latency_passed = None
    else:
        latency_passed = all(result["latency"]["passed"] for result in checked)
    report = {
        "device": device,
        "limit": args.limit,
        "seed": args.seed,
        "latency_mode": args.latency,
        "latency_passed": latency_passed,
        "suites": results,
        "passed": all(result["passed"] for result in results),
    }
    if latency_passed is False:
        if args.latency == "warn":
            print(
                "WARNING: latency budget exceeded (mode=warn, job stays green)",
                flush=True,
            )
        else:
            print("LATENCY REGRESSION: latency budget exceeded", flush=True)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"report -> {args.out}")
    if not report["passed"]:
        return 1
    if args.latency == "fail" and latency_passed is False:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
