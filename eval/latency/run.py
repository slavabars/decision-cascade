"""CLI замера латентности принятия решений.

In-process с разбивкой по фазам (encode / heads / decide):

    python -m eval.latency.run --n 30 --warmup 3

Через HTTP-эндпоинт (свой или внешний SystemOne-совместимый):

    python -m eval.latency.run --http http://127.0.0.1:8000/decisions --n 30
"""
from __future__ import annotations

import argparse
import json
from typing import Any

from eval.latency.benchmark import LatencyBenchmark, format_table, profile_phases
from src.sampler.questions import ChoiceSpec, NoulSpec, ScoreSpec

DEMO_STATE = "Stripe checkout failing for 3 days, we are losing sales"
DEMO_QUESTIONS = {
    "urgent": NoulSpec("Is this urgent?"),
    "team": ChoiceSpec("Which team should handle this?", ("billing", "technical")),
    "risk": ScoreSpec("Severity?", ("low", "medium", "high")),
}
HTTP_PAYLOAD: dict[str, Any] = {
    "state": DEMO_STATE,
    "questions": {
        "urgent": {"type": "noul", "instructions": "Is this urgent?"},
        "team": {
            "type": "choice",
            "instructions": "Which team should handle this?",
            "options": ["billing", "technical"],
        },
        "risk": {
            "type": "score",
            "instructions": "Severity?",
            "levels": ["low", "medium", "high"],
        },
    },
}


def _run_local(
    warmup: int,
    n: int,
    *,
    encoder_id: str | None = None,
    dim: int = 768,
    device: str = "auto",
    dtype: str = "float32",
    max_length: int | None = 256,
    batch_size: int = 16,
) -> dict[str, Any]:
    from src.device import check_dtype_device, resolve_device, resolve_dtype

    resolved_device = resolve_device(device)
    resolved_dtype = resolve_dtype(dtype)
    check_dtype_device(resolved_dtype, resolved_device)
    if encoder_id:
        from src.encoder.gemma import GemmaStateEncoder

        encoder = GemmaStateEncoder(
            model_id=encoder_id,
            feature_dim=dim,
            dtype=resolved_dtype,
            device=resolved_device,
            max_length=max_length,
            batch_size=batch_size,
        ).load()
        from api.server.service import DecisionService

        service = DecisionService(encoder=encoder, dim=dim)
    else:
        from api.server.service import DecisionService

        service = DecisionService(seed=0)
    phases = profile_phases(
        service, DEMO_STATE, DEMO_QUESTIONS, repeats=n, warmup=warmup
    )
    end_to_end = LatencyBenchmark(
        callable=lambda: service.decide(DEMO_STATE, DEMO_QUESTIONS)
    ).run(warmup=warmup, n=n)
    return {
        "mode": "local",
        "encoder": encoder_id or "hash",
        "device": resolved_device,
        "dtype": str(resolved_dtype),
        "phases": phases,
        "end_to_end": end_to_end,
    }


def _run_http(url: str, warmup: int, n: int, timeout: float) -> dict[str, Any]:
    import httpx

    def call() -> httpx.Response:
        response = httpx.post(url, json=HTTP_PAYLOAD, timeout=timeout)
        response.raise_for_status()
        return response

    report = LatencyBenchmark(callable=call).run(warmup=warmup, n=n)
    return {"mode": "http", "url": url, "end_to_end": report}


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--http", help="URL эндпоинта /decisions или /v1/systemone")
    parser.add_argument("--n", type=int, default=30, help="число замеров")
    parser.add_argument("--warmup", type=int, default=3, help="прогревочные прогоны")
    parser.add_argument("--timeout", type=float, default=60.0, help="таймаут HTTP, с")
    parser.add_argument(
        "--encoder",
        help="HF-id текстового энкодера для локального режима (иначе hash-энкодер)",
    )
    parser.add_argument("--dim", type=int, default=768, help="feature_dim энкодера")
    parser.add_argument(
        "--device",
        default="auto",
        help="auto|cpu|cuda|cuda:0|mps (DECISION_DEVICE-совместимо)",
    )
    parser.add_argument(
        "--dtype",
        default="float32",
        help="float32|float16|bfloat16 (DECISION_DTYPE-совместимо)",
    )
    parser.add_argument(
        "--json", action="store_true", help="выводить только JSON-отчёт"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.http:
        report = _run_http(args.http, args.warmup, args.n, args.timeout)
    else:
        report = _run_local(
            args.warmup,
            args.n,
            encoder_id=args.encoder,
            dim=args.dim,
            device=args.device,
            dtype=args.dtype,
        )

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    if "phases" in report:
        print(f"state: {DEMO_STATE!r}")
        print(format_table(report["phases"]))
        print()
    summary = report["end_to_end"]
    print(
        "end-to-end: "
        f"n={summary['count']} "
        f"mean={summary['mean']:.2f}ms "
        f"p50={summary['p50']:.2f}ms "
        f"p95={summary['p95']:.2f}ms "
        f"p99={summary['p99']:.2f}ms "
        f"throughput={summary['throughput_per_sec']:.1f}/s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
