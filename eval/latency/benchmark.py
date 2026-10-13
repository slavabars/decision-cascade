"""Бенчмарк латентности: замер p50/p95/p99 end-to-end прогонов."""
from __future__ import annotations

import statistics
import time
from collections.abc import Callable


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