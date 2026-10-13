"""Тесты генераторов датасетов (текст + изображения) — RED до реализации."""
import json

from eval.datasets.image_dataset import generate_images, write_manifest
from eval.datasets.text_dataset import (
    QUESTIONS,
    SPECS,
    generate_examples,
    write_jsonl,
)


def test_text_generate_count_and_schema():
    examples = generate_examples(n=50, seed=0)
    assert len(examples) == 50
    for ex in examples:
        assert set(ex) >= {"id", "state", "labels"}
        assert ex["state"]
        assert set(ex["labels"]) == set(QUESTIONS)
        for name, value in ex["labels"].items():
            spec = SPECS[name]
            if spec.__class__.__name__ == "NoulSpec":
                assert value in (0, 1)
            elif spec.__class__.__name__ == "ChoiceSpec":
                assert value in spec.options
            else:
                assert 0 <= value < len(spec.levels)


def test_text_generate_is_deterministic():
    a = generate_examples(n=20, seed=42)
    b = generate_examples(n=20, seed=42)
    assert a == b
    c = generate_examples(n=20, seed=7)
    assert a != c


def test_text_write_jsonl(tmp_path):
    path = tmp_path / "tickets.jsonl"
    examples = generate_examples(n=5, seed=1)
    write_jsonl(path, examples)
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    assert lines == examples


def test_images_generate_files_and_manifest(tmp_path):
    manifest = generate_images(n=8, out_dir=tmp_path, seed=0)
    pngs = sorted(tmp_path.glob("*.png"))
    assert len(pngs) == 8
    assert len(manifest) == 8
    for item in manifest:
        assert (tmp_path / item["file"]).exists()
        assert item["labels"]["color"] in {"red", "green", "blue", "yellow"}
        assert item["labels"]["shape"] in {"circle", "square", "triangle"}
        assert item["labels"]["count"] >= 1


def test_images_manifest_roundtrip(tmp_path):
    manifest = generate_images(n=4, out_dir=tmp_path, seed=3)
    path = tmp_path / "manifest.jsonl"
    write_manifest(path, manifest)
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    assert lines == manifest
