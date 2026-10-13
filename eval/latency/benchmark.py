"""Бенчмарк латентности: замер p50/p95/p99 end-to-end прогонов и по фазам."""
from __future__ import annotations

import statistics
import time
from collections.abc import Callable
from typing import TYPE_CHECKING

from src.sampler.parallel import sample_decisions
from src.sampler.questions import build_prompt

if TYPE_CHECKING:
    from api.server.service import DecisionService
    from src.sampler.questions import QuestionSpec

DEFAULT_COLUMNS = ("mean", "p50", "p95", "p99")


def summarize_latencies(samples: list[float]) -> dict[str, float]:
    ordered = sorted(samples)
    q = statistics.quantiles(ordered, n=100, method="inclusive")
    return {
        "count": len(ordered),
        "mean": float(statistics.mean(ordered)),
        "min": ordered[0],
        "p50": q[49],
        "p95": q[94],
        "p99": q[98],
        "max": ordered[-1],
    }


class LatencyBenchmark:
    def __init__(self, callable: Callable[[], object]):
        self._callable = callable

    def run(self, warmup: int = 3, n: int = 30) -> dict[str, float]:
        for _ in range(warmup):
            self._callable()
        samples: list[float] = []
        for _ in range(n):
            start = time.perf_counter()
            self._callable()
            samples.append((time.perf_counter() - start) * 1000.0)
        report = summarize_latencies(samples)
        report["throughput_per_sec"] = n / (sum(samples) / 1000.0)
        return report


def _time_call(fn: Callable[[], object], warmup: int, repeats: int) -> list[float]:
    for _ in range(warmup):
        fn()
    samples: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1000.0)
    return samples


def profile_phases(
    service: DecisionService,
    state: str,
    questions: dict[str, QuestionSpec],
    repeats: int = 20,
    warmup: int = 3,
) -> dict[str, dict[str, float]]:
    """Разбивает латентность in-process решения на фазы: encode, heads, decide.

    `encode` — прямой прогон энкодера (обходя кэш признаков сервиса), `heads` —
    прямой прогон голов, `decide` — полный путь `DecisionService.decide`,
    включая сборку промптов. В сумме `encode + heads` должны приближаться к
    `decide`; разница — накладные расходы.
    """
    prompts = [build_prompt(state, spec) for spec in questions.values()]
    features = service.encoder.encode(prompts)
    heads = service.get_heads(features.shape[-1], questions)

    return {
        "encode": summarize_latencies(
            _time_call(lambda: service.encoder.encode(prompts), warmup, repeats)
        ),
        "heads": summarize_latencies(
            _time_call(
                lambda: sample_decisions(heads, features), warmup, repeats
            )
        ),
        "decide": summarize_latencies(
            _time_call(lambda: service.decide(state, questions), warmup, repeats)
        ),
    }


def format_table(
    phases: dict[str, dict[str, float]],
    columns: tuple[str, ...] = DEFAULT_COLUMNS,
    unit: str = "ms",
) -> str:
    """Рендерит пофазный отчёт в текстовую таблицу для CLI/логов."""
    header = f"{'phase':<10}" + "".join(f"{col:>10}" for col in columns)
    lines = [header, "-" * len(header)]
    for name, summary in phases.items():
        cells = "".join(f"{summary.get(col, float('nan')):>10.2f}" for col in columns)
        lines.append(f"{name:<10}{cells}")
    counts = [int(s.get("count", 0)) for s in phases.values()]
    lines.append(f"({unit}, n>={min(counts, default=0)})")
    return "\n".join(lines)