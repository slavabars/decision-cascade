"""Русскоязычный набор: маппинг меток, вопросы и запись JSONL."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.datasets import ru_sentiment
from src.sampler.questions import ChoiceSpec, NoulSpec

ROOT = Path(__file__).resolve().parents[1]


def test_map_example_for_each_label():
    neutral = ru_sentiment.map_example("n", "нормально", 0)
    assert neutral["labels"] == {
        "sentiment": "neutral",
        "is_positive": 0,
        "is_negative": 0,
    }
    positive = ru_sentiment.map_example("p", "отлично", 1)
    assert positive["labels"]["is_positive"] == 1
    assert positive["labels"]["is_negative"] == 0
    negative = ru_sentiment.map_example("m", "ужасно", 2)
    assert negative["labels"]["is_negative"] == 1
    assert negative["labels"]["is_positive"] == 0


def test_map_example_rejects_unknown_label():
    with pytest.raises(ValueError, match="unknown sentiment"):
        ru_sentiment.map_example("x", "текст", 3)


def test_questions_match_labels():
    assert ru_sentiment.RU_LABELS == ("neutral", "positive", "negative")
    choice = ru_sentiment.RU_SPECS["sentiment"]
    assert isinstance(choice, ChoiceSpec)
    assert choice.options == ru_sentiment.RU_LABELS
    for name in ("is_positive", "is_negative"):
        assert isinstance(ru_sentiment.RU_SPECS[name], NoulSpec)


def test_label_distribution_counts():
    examples = [
        ru_sentiment.map_example("a", "1", 0),
        ru_sentiment.map_example("b", "2", 1),
        ru_sentiment.map_example("c", "3", 1),
        ru_sentiment.map_example("d", "4", 2),
    ]
    assert ru_sentiment.label_distribution(examples) == {
        "neutral": 1,
        "positive": 2,
        "negative": 1,
    }


def test_main_writes_jsonl(tmp_path, monkeypatch):
    examples = [
        ru_sentiment.map_example("a", "первый", 1),
        ru_sentiment.map_example("b", "второй", 2),
    ]
    monkeypatch.setattr(
        ru_sentiment,
        "load_ru_sentiment",
        lambda limit, seed=0: {"train": examples, "test": examples[:1]},
    )
    ru_sentiment.main(["--out-dir", str(tmp_path), "--limit", "10"])
    rows = [
        json.loads(line)
        for line in (tmp_path / "train.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [row["id"] for row in rows] == ["a", "b"]
    assert (tmp_path / "test.jsonl").is_file()


def test_revision_is_pinned():
    assert len(ru_sentiment.RU_REVISION) == 40
    assert ru_sentiment.RU_DATASET == "MonoHime/ru_sentiment_dataset"


def test_training_recipes_know_the_spec():
    train_heads = (ROOT / "training" / "recipe" / "train_heads.py").read_text(
        encoding="utf-8"
    )
    calibrate = (ROOT / "training" / "recipe" / "calibrate_temperature.py").read_text(
        encoding="utf-8"
    )
    for source in (train_heads, calibrate):
        assert '"ru_sentiment"' in source
        assert "RU_QUESTIONS" in source or "RU_SPECS" in source
