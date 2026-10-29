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
