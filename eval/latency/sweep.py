"""Бюджет латентности: батчинг, длина входа и dtype.

Отвечает на вопросы эксплуатации числами, а не ощущениями:

- сколько стоит один запрос при разной обрезке входа (``--max-length``);
- какой размер батча даёт максимум пропускной способности на ``/decisions/batch``;
- что даёт ``float16``/``bfloat16`` и где они вообще доступны.

Примеры:
    python -m eval.latency.sweep --encoder openai-community/gpt2 --dim 768 \\
        --device cpu --out data/latency/gpt2-cpu.json

    DECISION_DEVICE=cuda python -m eval.latency.sweep \\
        --encoder openai-community/gpt2 --dim 768 --kind causal \\
        --dtypes float32,float16,bfloat16
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from eval.latency.benchmark import LatencyBenchmark
from src.sampler.questions import ChoiceSpec, NoulSpec, ScoreSpec

QUESTIONS = {
    "urgent": NoulSpec("Is this urgent?"),
    "team": ChoiceSpec("Which team should handle this?", ("billing", "technical")),
    "risk": ScoreSpec("Severity?", ("low", "medium", "high")),
}

#: Разные состояния (а не одно, повторённое N раз): иначе кэш признаков по
#: промпту делает замер бессмысленным.
_STATES = (
    "Stripe checkout failing for {n} days, we are losing sales",
    "Клиент пишет, что не приходит письмо со сбросом пароля, уже {n} раз",
    "API отвечает 502 после деплоя, затронуто {n}% запросов",
    "Счёт вырос втрое, бухгалтерия просит объяснить списания за {n} дней",
    "Интеграция с CRM падает на импорте {n} контактов",
)


def make_states(count: int, offset: int = 0) -> list[str]:
    """Разные состояния: ``offset`` сдвигает нумерацию, чтобы повторы не попали
    в кэш признаков ``DecisionService`` (он кэширует по тексту промпта)."""
    return [
        _STATES[(offset + index) % len(_STATES)].format(n=offset + index + 1)
        for index in range(count)
    ]


def _service(
    encoder_id: str | None,
    dim: int,
    *,
    kind: str | None,
    device: str,
    dtype: str,
    max_length: int | None,
    batch_size: int,
):
    from api.server.service import DecisionService

    if not encoder_id:
        return DecisionService(seed=0), "hash"
    from src.device import check_dtype_device, resolve_device, resolve_dtype
    from src.encoder.factory import create_state_encoder

    resolved_device = resolve_device(device)
    resolved_dtype = resolve_dtype(dtype)
    check_dtype_device(resolved_dtype, resolved_device)
    encoder = create_state_encoder(
        encoder_id,
        dim,
        kind=kind,
        dtype=resolved_dtype,
        device=resolved_device,
        max_length=max_length,
        batch_size=batch_size,
    ).load()
    return (
        DecisionService(encoder=encoder, dim=dim),
        f"{encoder_id}/{resolved_dtype}@{resolved_device}",
    )


def measure_batch(
    service,
    *,
    count: int,
    repeats: int = 3,
    offset: int = 0,
) -> dict[str, float]:
    """Пропускная способность ``decide_many``: лучший из повторов, мс на состояние.

    Каждый повтор кодирует **новые** состояния: иначе второй и третий замеры
    попадают в кэш признаков сервиса и показывают задержку без энкодера.
    """
    if repeats < 1:
        raise ValueError(f"repeats must be >= 1, got {repeats}")
    if count < 1:
        raise ValueError(f"count must be >= 1, got {count}")
    times: list[float] = []
    for repeat in range(repeats):
        payload = make_states(count, offset=offset + repeat * count)
        started = time.perf_counter()
        service.decide_many(payload, QUESTIONS)
        times.append(time.perf_counter() - started)
    best = min(times)
    return {
        "states": count,
        "total_s": best,
        "per_state_ms": best * 1000.0 / count,
        "states_per_sec": count / best,
    }


def sweep_batch(
    encoder_id: str | None,
    dim: int,
    *,
    kind: str | None,
    device: str,
    dtype: str,
    max_length: int | None,
    batch_sizes: list[int],
    states: int,
    repeats: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for batch in batch_sizes:
        service, label = _service(
            encoder_id,
            dim,
            kind=kind,
            device=device,
            dtype=dtype,
            max_length=max_length,
            batch_size=batch,
        )
        result = measure_batch(service, count=states, repeats=repeats)
        rows.append({"batch": batch, "encoder": label, **result})
    return rows


def sweep_length(
    encoder_id: str | None,
    dim: int,
    *,
    kind: str | None,
    device: str,
    dtype: str,
    lengths: list[int | None],
    n: int,
    warmup: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for max_length in lengths:
        service, label = _service(
            encoder_id,
            dim,
            kind=kind,
            device=device,
            dtype=dtype,
            max_length=max_length,
            batch_size=16,
        )
        report = LatencyBenchmark(
            callable=_fresh_call(service)
        ).run(warmup=warmup, n=n)
        rows.append(
            {
                "max_length": max_length,
                "encoder": label,
                "p50_ms": report["p50"],
                "p95_ms": report["p95"],
                "mean_ms": report["mean"],
            }
        )
    return rows


#: Длинное состояние для свипа длины входа: на коротком тексте обрезка
#: ничего не меняет, и таблица была бы одинаковой при любом max_length.
_LONG_STATE = (
    "Клиент сообщает о сбоях оплаты, повторных списаниях и потерянных письмах. "
    "Затронуты несколько регионов, поддержка фиксирует рост обращений. "
) * 40


def payload_state() -> str:
    return _LONG_STATE


def _fresh_call(service):
    """Callable для ``LatencyBenchmark``: каждое измерение — новое состояние.

    ``LatencyBenchmark`` повторяет один и тот же вызов; с фиксированным текстом
    все замеры после первого шли бы из кэша признаков.
    """
    counter = {"index": 0}

    def call():
        index = counter["index"]
        counter["index"] += 1
        state = _LONG_STATE + f" Обращение №{index}."
        return service.decide(state, QUESTIONS)

    return call


def sweep_dtype(
    encoder_id: str | None,
    dim: int,
    *,
    kind: str | None,
    device: str,
    dtypes: list[str],
    n: int,
    warmup: int,
) -> list[dict[str, Any]]:
    """dtype-свип; недоступная комбинация попадает в отчёт как ``error``."""
    rows: list[dict[str, Any]] = []
    for dtype in dtypes:
        try:
            service, label = _service(
                encoder_id,
                dim,
                kind=kind,
                device=device,
                dtype=dtype,
                max_length=256,
                batch_size=16,
            )
            report = LatencyBenchmark(callable=_fresh_call(service)).run(
                warmup=warmup, n=n
            )
            rows.append(
                {
                    "dtype": dtype,
                    "encoder": label,
                    "p50_ms": report["p50"],
                    "p95_ms": report["p95"],
                    "error": None,
                }
            )
        except (RuntimeError, ValueError) as exc:
            rows.append(
                {
                    "dtype": dtype,
                    "encoder": f"{encoder_id}/{dtype}@{device}",
                    "p50_ms": None,
                    "p95_ms": None,
                    "error": str(exc),
                }
            )
    return rows


def render_batch(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| batch | состояний | мс/состояние | состояний/с |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['batch']} | {row['states']} | {row['per_state_ms']:.1f} | "
            f"{row['states_per_sec']:.1f} |"
        )
    return "\n".join(lines)


def render_length(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| max_length | p50, мс | p95, мс | mean, мс |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        length = "без обрезки" if row["max_length"] is None else row["max_length"]
        lines.append(
            f"| {length} | {row['p50_ms']:.1f} | {row['p95_ms']:.1f} | "
            f"{row['mean_ms']:.1f} |"
        )
    return "\n".join(lines)


def render_dtype(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| dtype | p50, мс | p95, мс | примечание |",
        "| --- | ---: | ---: | --- |",
    ]
    for row in rows:
        if row["error"]:
            lines.append(f"| {row['dtype']} | — | — | {row['error']} |")
        else:
            lines.append(
                f"| {row['dtype']} | {row['p50_ms']:.1f} | {row['p95_ms']:.1f} | — |"
            )
    return "\n".join(lines)


def _parse_lengths(raw: str) -> list[int | None]:
    values: list[int | None] = []
    for part in raw.split(","):
        part = part.strip().lower()
        if not part:
            continue
        values.append(None if part in {"none", "0", "off"} else int(part))
    return values


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder", default=None, help="HF-id или пусто (hash)")
    parser.add_argument("--dim", type=int, default=768)
    parser.add_argument("--kind", choices=["causal", "encoder"], default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--batch-sizes", default="1,8,16,32,64")
    parser.add_argument("--lengths", default="64,128,256")
    parser.add_argument(
        "--dtypes", default="", help="если задано — свип dtype (например float32,float16)"
    )
    parser.add_argument("--states", type=int, default=64)
    parser.add_argument("--n", type=int, default=20, help="замеров на конфигурацию")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out", type=Path, default=None, help="JSON с сырыми числами")
    parser.add_argument("--markdown", type=Path, default=None, help="таблицы в markdown")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = _parse_args(argv)
    batches = [int(part) for part in args.batch_sizes.split(",") if part.strip()]
    lengths = _parse_lengths(args.lengths)

    report: dict[str, Any] = {
        "encoder": args.encoder or "hash",
        "dim": args.dim,
        "kind": args.kind,
        "device": args.device,
        "dtype": args.dtype,
        "batch": sweep_batch(
            args.encoder,
            args.dim,
            kind=args.kind,
            device=args.device,
            dtype=args.dtype,
            max_length=256,
            batch_sizes=batches,
            states=args.states,
            repeats=args.repeats,
        ),
        "length": sweep_length(
            args.encoder,
            args.dim,
            kind=args.kind,
            device=args.device,
            dtype=args.dtype,
            lengths=lengths,
            n=args.n,
            warmup=args.warmup,
        ),
    }
    dtypes = [part.strip() for part in args.dtypes.split(",") if part.strip()]
    if dtypes:
        report["dtype_sweep"] = sweep_dtype(
            args.encoder,
            args.dim,
            kind=args.kind,
            device=args.device,
            dtypes=dtypes,
            n=args.n,
            warmup=args.warmup,
        )

    markdown = [
        f"# Латентность: {report['encoder']} ({args.device}, {args.dtype})",
        "",
        "## Батчинг",
        "",
        render_batch(report["batch"]),
        "",
        "## Длина входа",
        "",
        render_length(report["length"]),
    ]
    if "dtype_sweep" in report:
        markdown += ["", "## dtype", "", render_dtype(report["dtype_sweep"])]
    text = "\n".join(markdown) + "\n"
    print(text, flush=True)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"report -> {args.out}", flush=True)
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(text, encoding="utf-8")
        print(f"markdown -> {args.markdown}", flush=True)
    return report


if __name__ == "__main__":
    main()
