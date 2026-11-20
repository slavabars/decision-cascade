"""Журнал обратной связи: сериализация решений, чтение, компакция, рестарт."""
from __future__ import annotations

import json

import pytest

from src.feedback_store import (
    KIND_BENEFIT,
    KIND_RISK,
    SCHEMA_VERSION,
    FeedbackStore,
    decision_to_record,
    group_records,
    record_to_decision,
)
from src.schemas import ChoiceResult, NoulResult, ScoreResult

DECISIONS = [
    NoulResult(noul=0.73, confidence=0.46),
    ChoiceResult(
        choice="billing",
        probabilities={"billing": 0.7, "tech": 0.3},
        confidence=0.7,
    ),
    ScoreResult(
        score=0.8,
        probabilities={"low": 0.1, "high": 0.9},
        confidence=0.9,
        legend=("low", "high"),
    ),
]


@pytest.mark.parametrize("decision", DECISIONS)
def test_decision_roundtrip_keeps_type_and_values(decision):
    restored = record_to_decision(decision_to_record(decision))
    assert restored == decision


def test_record_to_decision_rejects_out_of_range():
    with pytest.raises(ValueError, match="noul"):
        record_to_decision({"type": "noul", "noul": 1.5, "confidence": 0.5})
    with pytest.raises(ValueError, match="confidence"):
        record_to_decision({"type": "noul", "noul": 0.5, "confidence": 2.0})
    with pytest.raises(ValueError, match="unknown decision type"):
        record_to_decision({"type": "bogus", "confidence": 0.5})
    with pytest.raises(TypeError, match="object"):
        record_to_decision(["noul"])


def test_store_appends_and_new_instance_loads(tmp_path):
    path = tmp_path / "nested" / "feedback.jsonl"

    store = FeedbackStore(path)
    store.append([{"kind": KIND_RISK, "question": "a", "correct": 1}])
    store.append([{"kind": KIND_RISK, "question": "b", "correct": 0}])

    assert path.is_file()
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [row["v"] for row in rows] == [SCHEMA_VERSION, SCHEMA_VERSION]
    assert FeedbackStore(path).load() == rows
    assert store.records_written == 2


def test_store_skips_broken_lines_and_warns(tmp_path, caplog):
    path = tmp_path / "feedback.jsonl"
    good = {"v": SCHEMA_VERSION, "kind": KIND_RISK, "question": "a", "correct": 1}
    path.write_text(
        "\n".join(
            [
                json.dumps(good),
                "{не json",
                json.dumps({**good, "v": 99}),
                json.dumps({**good, "kind": "unknown"}),
                json.dumps([1, 2, 3]),
                "",
            ]
        ),
        encoding="utf-8",
    )

    with caplog.at_level("WARNING"):
        records = FeedbackStore(path).load()

    assert records == [good]
    assert "пропущено 4" in caplog.text


def test_store_compacts_oldest_records(tmp_path):
    path = tmp_path / "feedback.jsonl"
    store = FeedbackStore(path, limit=3)
    for index in range(5):
        store.append([{"kind": KIND_RISK, "question": f"q{index}", "correct": 1}])

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [row["question"] for row in rows] == ["q2", "q3", "q4"]
    assert FeedbackStore(path).load() == rows
    # компакция атомарна: временный файл не остаётся
    assert sorted(item.name for item in tmp_path.iterdir()) == ["feedback.jsonl"]


def test_store_rejects_bad_kind_and_limit(tmp_path):
    store = FeedbackStore(tmp_path / "f.jsonl")
    with pytest.raises(ValueError, match="kind"):
        store.append([{"kind": "other"}])
    with pytest.raises(ValueError, match="limit"):
        FeedbackStore(tmp_path / "g.jsonl", limit=0)


def test_group_records_filters_by_kind():
    records = [
        {"kind": KIND_RISK, "question": "a"},
        {"kind": KIND_BENEFIT, "question": "b"},
        {"kind": KIND_RISK, "question": "c"},
    ]
    assert [row["question"] for row in group_records(records, KIND_RISK)] == ["a", "c"]
    assert [row["question"] for row in group_records(records, KIND_BENEFIT)] == ["b"]
