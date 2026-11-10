"""Свип латентности: батчинг, длина входа, dtype — на hash-энкодере, без сети."""
from __future__ import annotations

import json

import pytest

from eval.latency import sweep


def test_make_states_are_distinct():
    states = sweep.make_states(12)
    assert len(states) == 12
    assert len(set(states)) == 12


def test_measure_batch_reports_per_state():
    service, label = sweep._service(
        None, 64, kind=None, device="cpu", dtype="float32", max_length=64, batch_size=8
    )
    assert label == "hash"
    result = sweep.measure_batch(service, count=4, repeats=1)
    assert result["states"] == 4
    assert result["per_state_ms"] > 0
    assert result["states_per_sec"] > 0
    with pytest.raises(ValueError, match="repeats"):
        sweep.measure_batch(service, count=1, repeats=0)


def test_sweep_batch_rows():
    rows = sweep.sweep_batch(
        None,
        64,
        kind=None,
        device="cpu",
        dtype="float32",
        max_length=64,
        batch_sizes=[1, 8],
        states=4,
        repeats=1,
    )
    assert [row["batch"] for row in rows] == [1, 8]
    assert all(row["states"] == 4 for row in rows)
    table = sweep.render_batch(rows)
    assert table.startswith("| batch |")
    assert "| 8 |" in table


def test_sweep_length_and_render():
    rows = sweep.sweep_length(
        None,
        64,
        kind=None,
        device="cpu",
        dtype="float32",
        lengths=[16, 32],
        n=2,
        warmup=1,
    )
    assert [row["max_length"] for row in rows] == [16, 32]
    table = sweep.render_length(rows)
    assert "| max_length |" in table


def test_sweep_dtype_records_unavailable_as_error():
    rows = sweep.sweep_dtype(
        None,
        64,
        kind=None,
        device="cpu",
        dtypes=["float32", "float16"],
        n=3,
        warmup=0,
    )
    assert rows[0]["dtype"] == "float32"
    assert rows[0]["error"] is None
    # на CPU float16 обычно недоступен — это фиксируется, а не падает
    table = sweep.render_dtype(rows)
    assert "| float16 |" in table


def test_main_writes_json_and_markdown(tmp_path):
    out = tmp_path / "sweep.json"
    md = tmp_path / "sweep.md"
    report = sweep.main(
        [
            "--dim",
            "32",
            "--batch-sizes",
            "1,4",
            "--lengths",
            "16",
            "--states",
            "4",
            "--n",
            "2",
            "--warmup",
            "1",
            "--repeats",
            "1",
            "--out",
            str(out),
            "--markdown",
            str(md),
        ]
    )
    assert [row["batch"] for row in report["batch"]] == [1, 4]
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert saved["encoder"] == "hash"
    assert "## Батчинг" in md.read_text(encoding="utf-8")
