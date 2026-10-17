"""Тесты синтетических сценариев риск-оценщика (`eval/risk/simulate.py`)."""

from __future__ import annotations

import json

import pytest

from eval.risk.simulate import SCENARIOS, VARIANTS, main, scenario_report

#: Небольшой, но достаточный для качественных выводов конфиг (быстро на CPU).
KW = {"seeds": [0, 1], "n_val": 200, "n_test": 600, "noise": 0.25, "C": 1.0}


def _by_variant(rows: list[dict]) -> dict[str, dict]:
    return {row["variant"]: row for row in rows}


def test_scenario_report_is_deterministic() -> None:
    first = scenario_report(SCENARIOS["S2"], **KW)
    second = scenario_report(SCENARIOS["S2"], **KW)
    assert first == second


def test_all_variants_and_metric_ranges() -> None:
    report = scenario_report(SCENARIOS["S1"], **KW)
    assert {row["variant"] for row in report["in_dist"]} == set(VARIANTS)
    assert {row["variant"] for row in report["loo"]} == set(VARIANTS)
    for row in report["in_dist"] + report["loo"]:
        assert 0.0 <= row["aurc"] <= 1.0
        assert 0.0 <= row["eaurc"] <= 1.0
        assert 0.0 <= row["ece"]
        assert 0.0 <= row["brier"] <= 1.0
        assert 0.0 <= row["acc_at_0.7_coverage"] <= 1.0


def test_collision_makes_surrogate_inert() -> None:
    """S1: у всех вопросов одинаковый `n_classes` — суррогат ничего не различает."""
    report = scenario_report(SCENARIOS["S1"], **KW)
    in_dist = _by_variant(report["in_dist"])
    loo = _by_variant(report["loo"])
    assert abs(in_dist["+n_classes"]["aurc"] - in_dist["base(3)"]["aurc"]) < 1e-3
    assert abs(loo["+n_classes"]["aurc"] - loo["base(3)"]["aurc"]) < 1e-3
    # а явный свободный член вопросы всё равно различает
    assert in_dist["no_n_classes+question"]["aurc"] < in_dist["base(3)"]["aurc"] - 0.02


def test_distinct_classes_gain_in_dist_but_not_in_transfer() -> None:
    """S2: суррогат различает вопросы — прибыль в пуле, в переносе инертен."""
    report = scenario_report(SCENARIOS["S2"], **KW)
    in_dist = _by_variant(report["in_dist"])
    loo = _by_variant(report["loo"])
    assert in_dist["+n_classes"]["aurc"] < in_dist["base(3)"]["aurc"] - 0.03
    assert abs(loo["+n_classes"]["aurc"] - loo["base(3)"]["aurc"]) < 0.01


def test_drift_breaks_transfer() -> None:
    """S3/S6: связь признака с ошибкой дрейфует — `production` хуже `base(3)`."""
    for name in ("S3", "S6"):
        report = scenario_report(SCENARIOS[name], **KW)
        loo = _by_variant(report["loo"])
        assert loo["production"]["aurc"] > loo["base(3)"]["aurc"]
        assert SCENARIOS[name].drift


def test_shots_zero_equals_loo() -> None:
    """Точка `n=0` — это ровно LOO: собственных строк вопроса нет."""
    report = scenario_report(SCENARIOS["S4"], **KW, points=(0, 3))
    loo = _by_variant(report["loo"])
    zero = _by_variant([row for row in report["shots"] if row["shots"] == 0])
    for variant in VARIANTS:
        assert zero[variant]["aurc"] == pytest.approx(loo[variant]["aurc"])


def test_cli_writes_summary(tmp_path) -> None:
    out_dir = tmp_path / "sim"
    code = main(
        [
            "--scenarios",
            "S1",
            "S5",
            "--seeds",
            "0",
            "--n-val",
            "80",
            "--n-test",
            "200",
            "--out-dir",
            str(out_dir),
        ]
    )
    assert code == 0
    payload = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    # S5 — отчёт по S1/S4, поэтому S4 обязан посчитаться вместе с ним
    assert {"S1", "S4"} <= set(payload["reports"])
    assert "shots" in payload["reports"]["S4"]
    assert payload["config"]["seeds"] == [0]
    assert payload["config"]["variants"] == list(VARIANTS)
