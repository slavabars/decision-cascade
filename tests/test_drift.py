"""Эксперимент дрейфа домена: сплиты по темам/приору, отчёт, таблица."""
from __future__ import annotations

import pytest

from api.server.service import HashStateEncoder
from eval.datasets.public_dataset import map_example
from eval.drift import run as drift

DIM = 64


def _ag_news(n_per_topic: int) -> list[dict]:
    examples = []
    for topic in range(4):
        for index in range(n_per_topic):
            examples.append(
                map_example(
                    f"t{topic}-{index}",
                    f"topic {topic} story number {index} with words w{topic}{index}",
                    topic,
                )
            )
    return examples


def test_split_by_topic_returns_disjoint_parts():
    examples = _ag_news(2)
    inside, outside = drift.split_by_topic(examples, {0, 1})
    assert len(inside) == 4 and len(outside) == 4
    assert {drift.topic_index(ex) for ex in inside} == {0, 1}
    assert {drift.topic_index(ex) for ex in outside} == {2, 3}


def test_sample_by_prior_respects_shares():
    examples = _ag_news(100)  # по 100 примеров на тему
    picked = drift.sample_by_prior(examples, [0.4, 0.4, 0.1, 0.1], 100, seed=0)
    counts = {topic: 0 for topic in range(4)}
    for example in picked:
        counts[drift.topic_index(example)] += 1
    assert counts[0] == 40 and counts[1] == 40
    assert counts[2] == 10 and counts[3] == 10
    # выборка не выдумывает примеры сверх доступных
    tight = drift.sample_by_prior(_ag_news(10), [0.9, 0.0, 0.1, 0.0], 100, seed=0)
    assert len(tight) == 10 + 10


def test_sample_by_prior_is_deterministic():
    examples = _ag_news(5)
    first = drift.sample_by_prior(examples, [0.25] * 4, 20, seed=3)
    second = drift.sample_by_prior(examples, [0.25] * 4, 20, seed=3)
    assert [ex["id"] for ex in first] == [ex["id"] for ex in second]


def test_run_experiment_label_shift(monkeypatch):
    splits = {"train": _ag_news(30), "test": _ag_news(10)}
    monkeypatch.setattr(drift, "load_ag_news", lambda limit: splits)
    report = drift.run_experiment(
        limit=120,
        old_topics={0, 1},
        sizes=[20],
        mode="label-shift",
        prior=[0.4, 0.4, 0.1, 0.1],
        encoder=HashStateEncoder(DIM),
        encoder_name="hash",
        train_steps=50,
        steps=20,
        lr=0.05,
        replay=1.0,
        seed=0,
    )
    assert report["mode"] == "label-shift"
    assert report["dataset"] == "ag_news"
    assert set(report["baseline"]) == {"old", "new"}
    run = report["runs"][0]
    assert run["k"] == 20
    assert 0.0 <= run["new"]["accuracy"] <= 1.0
    assert run["gate"]["accepted"] in (True, False)
    assert report["n"]["old_train"] > 0 and report["n"]["new_train"] > 0


def test_run_experiment_new_topics(monkeypatch):
    splits = {"train": _ag_news(20), "test": _ag_news(5)}
    monkeypatch.setattr(drift, "load_ag_news", lambda limit: splits)
    report = drift.run_experiment(
        limit=80,
        old_topics={0, 1},
        sizes=[10, 100],
        mode="new-topics",
        encoder=HashStateEncoder(DIM),
        encoder_name="hash",
        train_steps=30,
        steps=10,
        lr=0.05,
        seed=0,
    )
    assert report["old_topics"] == ["World", "Sports"]
    assert report["new_topics"] == ["Business", "Sci/Tech"]
    # K больше доступного урезается, а не падает
    assert [run["k"] for run in report["runs"]] == [10, 40]


def test_run_experiment_rejects_unknown_mode(monkeypatch):
    monkeypatch.setattr(drift, "load_ag_news", lambda limit: {"train": [], "test": []})
    with pytest.raises(ValueError, match="unknown mode"):
        drift.run_experiment(
            limit=10,
            old_topics={0},
            sizes=[1],
            mode="nope",
            encoder=HashStateEncoder(DIM),
            encoder_name="hash",
        )


def test_run_experiment_rejects_empty_split(monkeypatch):
    splits = {"train": _ag_news(2), "test": _ag_news(1)}
    monkeypatch.setattr(drift, "load_ag_news", lambda limit: splits)
    with pytest.raises(ValueError, match="не хватает данных"):
        drift.run_experiment(
            limit=8,
            old_topics={0, 1, 2, 3},
            sizes=[1],
            mode="new-topics",
            encoder=HashStateEncoder(DIM),
            encoder_name="hash",
            train_steps=1,
            steps=1,
        )


def test_render_markdown_has_table_rows(monkeypatch):
    splits = {"train": _ag_news(20), "test": _ag_news(5)}
    monkeypatch.setattr(drift, "load_ag_news", lambda limit: splits)
    report = drift.run_experiment(
        limit=80,
        old_topics={0, 1},
        sizes=[10],
        encoder=HashStateEncoder(DIM),
        encoder_name="hash",
        train_steps=30,
        steps=10,
        lr=0.05,
    )
    markdown = drift.render_markdown(report)
    assert "| K |" in markdown
    assert "| 10 |" in markdown
    assert "ag_news" in markdown


def test_main_writes_json_and_markdown(monkeypatch, tmp_path):
    splits = {"train": _ag_news(20), "test": _ag_news(5)}
    monkeypatch.setattr(drift, "load_ag_news", lambda limit: splits)
    out = tmp_path / "report.json"
    md = tmp_path / "report.md"
    report = drift.main(
        [
            "--limit",
            "80",
            "--sizes",
            "10",
            "--train-steps",
            "20",
            "--steps",
            "5",
            "--out",
            str(out),
            "--markdown",
            str(md),
        ],
        encoder=HashStateEncoder(DIM),
    )
    assert report["runs"][0]["k"] == 10
    assert out.is_file() and md.is_file()
    assert "| K |" in md.read_text(encoding="utf-8")


def test_run_experiment_control_row(monkeypatch):
    splits = {"train": _ag_news(30), "test": _ag_news(10)}
    monkeypatch.setattr(drift, "load_ag_news", lambda limit: splits)
    report = drift.run_experiment(
        limit=120,
        old_topics={0, 1},
        sizes=[10],
        encoder=HashStateEncoder(DIM),
        encoder_name="hash",
        train_steps=20,
        steps=5,
        lr=0.05,
        control=True,
        seed=0,
    )
    assert [run["k"] for run in report["runs"]] == [0, 10]
    assert report["control"] is True
    assert "accuracy" in report["runs"][0]["new"]


def test_retrain_accepts_replay_only(tmp_path):
    from training.feedback.run import retrain

    old = _ag_news(5)
    heads = drift.train_base(
        old, HashStateEncoder(DIM), steps=5, lr=0.05, seed=0
    )
    trained = retrain(
        heads, drift.PUBLIC_SPECS, HashStateEncoder(DIM), [], old, steps=2, lr=0.05
    )
    assert trained is not heads
