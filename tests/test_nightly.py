"""Ночной регресс: порог, отчёт и коды выхода без загрузки моделей."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval import nightly

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "eval" / "nightly_baseline.json"


def test_shipped_baseline_covers_every_suite():
    payload = json.loads(BASELINE.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "1"
    for suite in nightly.SUITES:
        entry = payload[suite]
        assert entry["pack"]
        assert entry["n"] > 0
        assert 0.0 <= entry["overall_accuracy"] <= 1.0
        assert 0.0 < entry["tolerance"] < 1.0


def test_baseline_packs_are_named_like_release_artifacts():
    payload = json.loads(BASELINE.read_text(encoding="utf-8"))
    for suite in nightly.SUITES:
        pack = payload[suite]["pack"]
        # веса публикуются как <pack>/heads.pt + metadata.json в *-weights.tar.gz
        assert not pack.endswith(".pt")
        assert "/" not in pack


@pytest.mark.parametrize(
    ("accuracy", "expected", "allowed", "passed"),
    [
        (0.88, 0.88, 0.01, True),
        (0.87, 0.88, 0.01, True),   # ровно на границе допуска
        (0.8699, 0.88, 0.01, False),  # падение больше 1 п.п.
        (0.90, 0.88, 0.01, True),   # рост — не регресс
    ],
)
def test_within_tolerance(accuracy, expected, allowed, passed):
    assert nightly.within_tolerance(accuracy, expected, allowed) is passed


def _fake_check(passed: bool):
    def check(suite, entry, packs_root, **kwargs):
        return {
            "suite": suite,
            "pack": entry["pack"],
            "n": entry["n"],
            "accuracy": entry["overall_accuracy"] - (0.0 if passed else 0.02),
            "baseline": entry["overall_accuracy"],
            "delta": 0.0 if passed else -0.02,
            "tolerance": entry["tolerance"],
            "passed": passed,
            "questions": {},
            "latency_ms": {},
        }

    return check


def test_main_returns_zero_when_all_suites_pass(tmp_path, monkeypatch):
    monkeypatch.setattr(nightly, "check_suite", _fake_check(True))
    out = tmp_path / "report.json"
    code = nightly.main(["--suite", "all", "--out", str(out)])
    assert code == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["passed"] is True
    assert [item["suite"] for item in report["suites"]] == list(nightly.SUITES)


def test_main_fails_on_regression(tmp_path, monkeypatch):
    monkeypatch.setattr(nightly, "check_suite", _fake_check(False))
    out = tmp_path / "report.json"
    code = nightly.main(["--suite", "ag_news", "--out", str(out)])
    assert code == 1
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["passed"] is False
    assert report["suites"][0]["delta"] == -0.02


def test_main_rejects_unknown_suite(tmp_path):
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({"ag_news": {}}), encoding="utf-8")
    with pytest.raises(SystemExit):
        nightly.main(["--suite", "all", "--baseline", str(baseline)])


# --- бюджет латентности ------------------------------------------------------


def test_baseline_has_latency_budget():
    payload = json.loads(BASELINE.read_text(encoding="utf-8"))
    for suite in nightly.SUITES:
        budget = payload[suite]["latency"]
        assert 0 < budget["p50_ms"] < budget["p95_ms"]
        assert budget["tolerance"] >= 1.0


def test_check_latency_without_budget_or_metrics():
    entry = {"latency": None}
    assert nightly.check_latency(entry, {"p50": 10.0, "p95": 20.0}) is None
    entry = {"latency": {"p50_ms": 100, "p95_ms": 200, "tolerance": 1.25}}
    assert nightly.check_latency(entry, {}) is None


def test_check_latency_respects_tolerance():
    entry = {"latency": {"p50_ms": 100, "p95_ms": 200, "tolerance": 1.5}}
    within = nightly.check_latency(entry, {"p50": 120.0, "p95": 280.0})
    assert within["passed"] is True
    assert within["checks"]["p95_ms"]["limit_ms"] == pytest.approx(300.0)
    slow = nightly.check_latency(entry, {"p50": 120.0, "p95": 301.0})
    assert slow["passed"] is False


def _fake_check_latency(latency_passed: bool):
    def check(suite, entry, packs_root, **kwargs):
        return {
            "suite": suite,
            "pack": entry["pack"],
            "n": entry["n"],
            "accuracy": entry["overall_accuracy"],
            "baseline": entry["overall_accuracy"],
            "delta": 0.0,
            "tolerance": entry["tolerance"],
            "passed": True,
            "questions": {},
            "latency_ms": {"p50": 10.0, "p95": 20.0},
            "latency": {
                "checks": {
                    "p95_ms": {
                        "value_ms": 20.0 if latency_passed else 10_000.0,
                        "limit_ms": 100.0,
                        "passed": latency_passed,
                    }
                },
                "passed": latency_passed,
            },
        }

    return check


def test_latency_fail_mode_returns_nonzero(tmp_path, monkeypatch):
    monkeypatch.setattr(nightly, "check_suite", _fake_check_latency(False))
    out = tmp_path / "report.json"
    code = nightly.main(
        ["--suite", "ag_news", "--latency", "fail", "--out", str(out)]
    )
    assert code == 1
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["passed"] is True
    assert report["latency_passed"] is False
    assert report["latency_mode"] == "fail"


def test_latency_warn_mode_keeps_job_green(tmp_path, monkeypatch):
    monkeypatch.setattr(nightly, "check_suite", _fake_check_latency(False))
    out = tmp_path / "report.json"
    code = nightly.main(
        ["--suite", "ag_news", "--latency", "warn", "--out", str(out)]
    )
    assert code == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["latency_passed"] is False


def test_latency_off_mode_reports_none(tmp_path, monkeypatch):
    monkeypatch.setattr(nightly, "check_suite", _fake_check_latency(False))
    out = tmp_path / "report.json"
    code = nightly.main(
        ["--suite", "ag_news", "--latency", "off", "--out", str(out)]
    )
    assert code == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["latency_passed"] is None
