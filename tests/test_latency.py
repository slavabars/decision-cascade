"""Тесты latency-бенчмарка — написаны до реализации (TDD RED)."""

from eval.latency.benchmark import LatencyBenchmark, summarize_latencies


def test_summarize_latencies_moments():
    samples = [1.0, 2.0, 3.0, 4.0, 5.0]
    report = summarize_latencies(samples)
    assert report["count"] == 5
    assert round(report["mean"], 2) == 3.0
    assert 0 < report["p50"] <= report["p95"] <= report["p99"]


def test_benchmark_returns_report():
    from fastapi.testclient import TestClient

    from api.server.main import app

    payload = {
        "state": "stripe failing for 3 days",
        "questions": {
            "u": {"type": "noul", "instructions": "urgent?"},
            "c": {
                "type": "choice",
                "instructions": "who?",
                "options": ["billing", "sales"],
            },
        },
    }
    with TestClient(app) as tc:
        bench = LatencyBenchmark(callable=lambda: tc.post("/decisions", json=payload))
        report = bench.run(warmup=1, n=5)
    assert report["count"] == 5
    assert report["p50"] > 0
    assert report["throughput_per_sec"] > 0