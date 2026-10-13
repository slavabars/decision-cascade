"""Тесты latency-бенчмарка — написаны до реализации (TDD RED)."""

from eval.latency.benchmark import (
    LatencyBenchmark,
    format_table,
    profile_phases,
    summarize_latencies,
)
from src.sampler.questions import ChoiceSpec, NoulSpec


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


def test_profile_phases_reports_components():
    from api.server.service import DecisionService

    svc = DecisionService(seed=0)
    questions = {
        "u": NoulSpec("urgent?"),
        "c": ChoiceSpec("who?", ("billing", "sales")),
    }
    report = profile_phases(
        svc, "stripe failing for 3 days", questions, repeats=3, warmup=1
    )
    assert set(report) == {"encode", "heads", "decide"}
    for summary in report.values():
        assert {"count", "mean", "p50", "p95", "p99"} <= set(summary)
        assert summary["mean"] >= 0.0


def test_profile_phases_reuse_builds_heads_once():
    from api.server.service import DecisionService

    svc = DecisionService(seed=0)
    questions = {"u": NoulSpec("urgent?")}
    profile_phases(svc, "state", questions, repeats=2, warmup=1)
    assert svc.get_heads(svc.dim, questions).dim > 0


def test_format_table_renders_rows():
    report = {
        "encode": {"mean": 1.0, "p50": 1.0, "p95": 1.5, "p99": 1.9},
        "decide": {"mean": 2.0, "p50": 2.0, "p95": 3.0, "p99": 4.0},
    }
    text = format_table(report)
    assert "phase" in text
    assert "p50" in text
    assert "encode" in text and "decide" in text


def test_cli_local_json_output(capsys):
    import json

    from eval.latency.run import main

    assert main(["--n", "2", "--warmup", "1", "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["mode"] == "local"
    assert "encode" in report["phases"]
    assert report["end_to_end"]["count"] == 2