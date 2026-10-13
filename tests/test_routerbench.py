"""Тесты публичного адаптера RouterBench и его связки с CLI — без сети."""
import math

import pytest

import eval.datasets.routerbench as rb
from eval.competitors.run import _specs as competitors_specs
from eval.competitors.run import _threshold_grid
from eval.report.run import _specs as report_specs
from src.sampler.questions import ChoiceSpec, NoulSpec, ScoreSpec


def test_routerbench_spec_types_and_options():
    assert isinstance(rb.ROUTERBENCH_SPECS["escalate"], NoulSpec)
    assert isinstance(rb.ROUTERBENCH_SPECS["route"], ChoiceSpec)
    assert isinstance(rb.ROUTERBENCH_SPECS["benefit"], ScoreSpec)
    assert list(rb.ROUTERBENCH_SPECS["route"].options) == list(rb.ROUTER_TIERS)
    assert list(rb.ROUTERBENCH_SPECS["benefit"].levels) == ["none", "some", "large"]
    assert rb.ROUTER_TIERS == ("cheap", "mid", "strong")


def test_specs_wiring_in_clis():
    cli_specs = competitors_specs("routerbench")
    assert set(cli_specs) == set(rb.ROUTERBENCH_QUESTIONS)
    report = report_specs("routerbench")
    assert set(report) == set(rb.ROUTERBENCH_SPECS)


def test_parse_prompt_joins_string_messages():
    prompt = "['system instruction', 'user question']"
    assert rb.parse_prompt(prompt) == "system instruction\n\nuser question"


def test_parse_prompt_handles_message_dicts_and_list_input():
    prompt = [{"role": "system", "content": "be nice"}, {"content": "hello"}]
    assert rb.parse_prompt(prompt) == "be nice\n\nhello"


def test_parse_prompt_returns_raw_when_unparseable():
    assert rb.parse_prompt("not a python literal [") == "not a python literal ["


def test_coerce_score_handles_nan_and_none():
    assert rb._coerce_score(None) == 0.0
    assert rb._coerce_score("oops") == 0.0
    assert math.isnan(float("nan"))
    assert rb._coerce_score(float("nan")) == 0.0
    assert rb._coerce_score(0.75) == 0.75


def _scores(cheap: float, mid: float, strong: float) -> dict[str, float]:
    return {"cheap": cheap, "mid": mid, "strong": strong}


def test_map_example_escalate_and_route_and_benefit():
    ex = rb.map_example("id-1", "['x']", _scores(0.0, 0.5, 1.0), eval_name="gsm8k")
    assert ex["id"] == "id-1"
    assert ex["state"] == "x"
    assert ex["labels"]["escalate"] == 1  # cheap не решил -> эскалируем
    assert ex["labels"]["route"] == "strong"
    assert ex["labels"]["benefit"] == 2  # gain = 1.0 -> large
    assert ex["metadata"]["eval_name"] == "gsm8k"


def test_map_example_solved_by_cheap():
    ex = rb.map_example("id-2", "x", _scores(1.0, 1.0, 1.0))
    assert ex["labels"]["escalate"] == 0
    assert ex["labels"]["route"] == "cheap"  # при равенстве — дешевле
    assert ex["labels"]["benefit"] == 0


def test_route_label_tie_break_prefers_cheaper():
    assert rb._route_label(_scores(1.0, 1.0, 1.0)) == "cheap"
    assert rb._route_label(_scores(0.0, 1.0, 1.0)) == "mid"
    assert rb._route_label(_scores(0.0, 0.0, 0.5)) == "strong"


def test_benefit_label_buckets():
    assert [rb._benefit_label(g) for g in (-1.0, 0.0, 0.25, 0.5, 0.75, 1.0)] == [
        0,
        0,
        1,
        1,
        2,
        2,
    ]


def test_escalate_label_boundary():
    assert rb._escalate_label(0.99) == 1
    assert rb._escalate_label(1.0) == 0


class _FakeFrame:
    def __init__(self, rows: list[dict]):
        self._rows = rows

    def to_dict(self, orient: str) -> list[dict]:
        assert orient == "records"
        return self._rows


def _fake_rows(n: int) -> list[dict]:
    rows = []
    for i in range(n):
        rows.append(
            {
                "prompt": f"['prompt {i}']",
                "eval_name": "gsm8k",
                rb.CHEAP_MODEL: float(i % 2),
                rb.MID_MODEL: 0.5,
                rb.STRONG_MODEL: 1.0 if i % 3 == 0 else 0.5,
            }
        )
    return rows


def _patch_frame(monkeypatch, rows):
    monkeypatch.setattr(rb, "_read_frame", lambda shot: _FakeFrame(rows))


def test_load_routerbench_splits_and_limit(monkeypatch):
    _patch_frame(monkeypatch, _fake_rows(40))
    splits = rb.load_routerbench(limit=8, seed=0)
    assert len(splits["train"]) == 8
    assert len(splits["test"]) == 2  # limit // 4
    expected = {"escalate", "route", "benefit"}
    assert all(ex["labels"].keys() == expected for ex in splits["train"])


def test_load_routerbench_is_deterministic(monkeypatch):
    _patch_frame(monkeypatch, _fake_rows(30))
    first = rb.load_routerbench(limit=9, seed=1)
    second = rb.load_routerbench(limit=9, seed=1)
    assert [ex["id"] for ex in first["test"]] == [ex["id"] for ex in second["test"]]


def test_load_routerbench_rejects_bad_shot(monkeypatch):
    _patch_frame(monkeypatch, _fake_rows(4))
    with pytest.raises(ValueError):
        rb.load_routerbench(shot="7shot")


def test_load_routerbench_rejects_bad_test_fraction(monkeypatch):
    _patch_frame(monkeypatch, _fake_rows(4))
    with pytest.raises(ValueError):
        rb.load_routerbench(test_fraction=1.5)


def test_threshold_grid_covers_zero_to_one():
    grid = _threshold_grid(11)
    assert grid[0] == 0.0
    assert grid[-1] == 1.0
    assert len(grid) == 11
    assert _threshold_grid(1) == [0.0, 1.0]  # минимум две точки
