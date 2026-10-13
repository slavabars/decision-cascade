"""Тесты датасет-сплитов, multi-split бенчмарка и рецепта обучения — RED."""
import hashlib
import json

import pytest

from api.server.service import HashStateEncoder
from eval.benchmark.core import run_benchmark
from eval.datasets.text_dataset import (
    SPECS,
    generate_examples,
    split_examples,
    write_jsonl,
)


def test_split_examples_sizes_and_disjoint():
    examples = generate_examples(n=100, seed=0)
    splits = split_examples(examples, val_fraction=0.2, test_fraction=0.1, seed=1)
    assert set(splits) == {"train", "val", "test"}
    assert len(splits["test"]) == 10
    assert len(splits["val"]) == 20
    assert len(splits["train"]) == 70
    ids = [ex["id"] for part in splits.values() for ex in part]
    assert len(ids) == len(set(ids)) == 100


def test_split_examples_is_deterministic():
    examples = generate_examples(n=50, seed=3)
    a = split_examples(examples, val_fraction=0.2, test_fraction=0.2, seed=7)
    b = split_examples(examples, val_fraction=0.2, test_fraction=0.2, seed=7)
    assert a == b


def test_split_examples_rejects_bad_fractions():
    examples = generate_examples(n=10, seed=0)
    with pytest.raises(ValueError):
        split_examples(examples, val_fraction=0.6, test_fraction=0.5)
    with pytest.raises(ValueError):
        split_examples(examples, val_fraction=-0.1, test_fraction=0.1)


def test_run_benchmark_with_splits_reports_held_out():
    examples = generate_examples(n=600, seed=0)
    splits = split_examples(examples, val_fraction=0.2, test_fraction=0.2, seed=1)
    report = run_benchmark(
        SPECS, examples, HashStateEncoder(64), train_steps=300, seed=0, splits=splits
    )
    assert set(report["splits"]) == {"train", "val", "test"}
    assert report["splits"]["test"]["n"] == len(splits["test"])
    assert report["splits"]["test"]["overall_accuracy"] == report["overall_accuracy"]
    assert report["questions"] == report["splits"]["test"]["questions"]
    assert report["overall_accuracy"] > 0.6


def test_train_benchmark_returns_reusable_heads():
    from eval.benchmark.core import evaluate_benchmark, train_benchmark

    examples = generate_examples(n=80, seed=0)
    heads, report = train_benchmark(
        SPECS, examples, HashStateEncoder(64), train_steps=50, seed=0
    )
    assert report["n"] == 80
    again = evaluate_benchmark(SPECS, examples, HashStateEncoder(64), heads)
    assert again["overall_accuracy"] == report["overall_accuracy"]


def test_recipe_writes_checkpoint_and_metadata(tmp_path):
    from training.recipe import train_heads

    dataset = tmp_path / "data.jsonl"
    examples = generate_examples(n=120, seed=0)
    write_jsonl(dataset, examples)
    out_dir = tmp_path / "ckpt"

    train_heads.main(
        [
            "--dataset",
            str(dataset),
            "--out-dir",
            str(out_dir),
            "--encoder",
            "hash",
            "--dim",
            "64",
            "--train-steps",
            "50",
            "--seed",
            "0",
        ]
    )

    ckpt = out_dir / "heads.pt"
    meta = out_dir / "metadata.json"
    assert ckpt.exists() and meta.exists()
    payload = json.loads(meta.read_text(encoding="utf-8"))
    assert payload["encoder"] == "hash"
    assert payload["feature_dim"] == 64
    assert payload["dataset"]["sha256"] == hashlib.sha256(
        dataset.read_bytes()
    ).hexdigest()
    assert payload["dataset"]["n"] == 120
    assert set(payload["splits"]) == {"train", "val", "test"}
    assert set(payload["metrics"]) == {"train", "val", "test"}
    for metrics in payload["metrics"].values():
        assert 0.0 <= metrics["overall_accuracy"] <= 1.0
    assert set(payload["questions"]) == set(SPECS)
    # Копия под канонической подписью — drop-in для DECISION_CHECKPOINT_DIR.
    from src.sampler.questions import questions_signature

    signature = questions_signature(SPECS)
    assert payload["questions_signature"] == signature
    assert (out_dir / f"{signature}.pt").exists()


def test_recipe_generates_dataset_when_missing(tmp_path):
    from training.recipe import train_heads

    out_dir = tmp_path / "ckpt"
    train_heads.main(
        [
            "--generate",
            "60",
            "--out-dir",
            str(out_dir),
            "--train-steps",
            "20",
            "--seed",
            "0",
        ]
    )
    assert (out_dir / "heads.pt").exists()
    assert (out_dir / "dataset.jsonl").exists()


def test_recipe_public_specs_uses_ag_news(tmp_path, monkeypatch):
    from eval.datasets import public_dataset
    from training.recipe import train_heads

    def fake_ag_news(limit=None):
        labels = {"topic": "World", "business": 0, "tech_level": 0}
        return {
            "train": [
                {"id": f"t{i}", "state": f"news {i}", "labels": labels}
                for i in range(40)
            ],
            "test": [
                {"id": f"e{i}", "state": f"news test {i}", "labels": labels}
                for i in range(10)
            ],
        }

    monkeypatch.setattr(public_dataset, "load_ag_news", fake_ag_news)

    out_dir = tmp_path / "ckpt"
    meta = train_heads.main(
        [
            "--specs",
            "public",
            "--limit",
            "40",
            "--out-dir",
            str(out_dir),
            "--encoder",
            "hash",
            "--dim",
            "32",
            "--train-steps",
            "5",
            "--seed",
            "0",
        ]
    )

    assert meta["specs"] == "public"
    assert meta["dataset_name"] == "ag_news"
    assert set(meta["questions"]) == {"topic", "business", "tech_level"}
    assert meta["splits"] == {"train": 32, "val": 8, "test": 10}
    assert (out_dir / "dataset.jsonl").exists()
