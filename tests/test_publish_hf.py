"""Публикация паков на Hugging Face: манифест, карточка, staging, гейты."""
from __future__ import annotations

import json

import pytest

from training.recipe import publish_hf

DIM = 8


def _pack(root, name: str, *, encoder: str = "openai-community/gpt2", dim: int = DIM):
    pack = root / name
    pack.mkdir(parents=True)
    (pack / "heads.pt").write_bytes(b"not-a-real-checkpoint")
    (pack / "temperature.json").write_text("{}", encoding="utf-8")
    (pack / "dataset.jsonl").write_text("{}\n", encoding="utf-8")
    (pack / "metadata.json").write_text(
        json.dumps(
            {
                "schema_version": "1",
                "encoder": encoder,
                "feature_dim": dim,
                "questions_signature": f"{abs(hash(name)):016x}"[:16],
                "dataset_name": "ag_news",
                "splits": {"train": 10, "val": 2, "test": 3},
                "questions": {"topic": {"type": "choice"}},
                "metrics": {
                    "test": {
                        "overall_accuracy": 0.75,
                        "questions": {
                            "topic": {
                                "accuracy": 0.75,
                                "brier": 0.2,
                                "ece": 0.05,
                            }
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return pack


def test_manifest_has_hashes_metrics_and_no_datasets(tmp_path):
    _pack(tmp_path, "public-gpt2")

    manifest = publish_hf.build_manifest(tmp_path, version="1.8.0", commit="abc")

    assert manifest["library_version"] == "1.8.0"
    assert manifest["git_commit"] == "abc"
    pack = manifest["packs"][0]
    assert pack["name"] == "public-gpt2"
    assert set(pack["files"]) == {"heads.pt", "metadata.json", "temperature.json"}
    assert pack["files"]["heads.pt"]["bytes"] == len(b"not-a-real-checkpoint")
    assert pack["metrics"]["test"]["overall_accuracy"] == 0.75
    assert pack["dataset_source"] == "fancyzhx/ag_news"
    assert pack["dataset_license"] == "unknown"


def test_manifest_excludes_named_packs(tmp_path):
    _pack(tmp_path, "public-gpt2")
    _pack(tmp_path, "chess")

    manifest = publish_hf.build_manifest(tmp_path, excludes=("chess",))

    assert [pack["name"] for pack in manifest["packs"]] == ["public-gpt2"]


def test_manifest_rejects_pack_without_heads(tmp_path):
    pack = _pack(tmp_path, "public-gpt2")
    (pack / "heads.pt").unlink()

    with pytest.raises(ValueError, match="heads.pt"):
        publish_hf.build_manifest(tmp_path)


def test_manifest_rejects_bad_feature_dim(tmp_path):
    pack = _pack(tmp_path, "public-gpt2")
    metadata = json.loads((pack / "metadata.json").read_text(encoding="utf-8"))
    metadata["feature_dim"] = "768"
    (pack / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="feature_dim"):
        publish_hf.build_manifest(tmp_path)


def test_card_describes_packs_metrics_and_not_transformers(tmp_path):
    _pack(tmp_path, "public-gpt2")
    manifest = publish_hf.build_manifest(tmp_path, version="1.8.0")

    card = publish_hf.render_card(
        manifest, repo="slavabars/decision-cascade-packs", revision="v1.8.0"
    )

    assert "license: apache-2.0" in card
    assert "not a `transformers` model" in card
    assert 'pip install decision-cascade' in card
    assert 'revision="v1.8.0"' in card
    assert "| `public-gpt2` | `openai-community/gpt2` | 8 |" in card
    assert "0.750" in card
    assert "fancyzhx/ag_news" in card
    assert "license: unknown" in card


def test_stage_copies_only_service_files_and_writes_manifest(tmp_path):
    _pack(tmp_path / "packs", "public-gpt2")
    manifest = publish_hf.build_manifest(tmp_path / "packs", version="1.8.0")
    out = tmp_path / "stage"

    publish_hf.stage(tmp_path / "packs", out, manifest, revision="v1.8.0")

    assert (out / "manifest.json").is_file()
    assert (out / "README.md").is_file()
    files = sorted(path.name for path in (out / "packs" / "public-gpt2").iterdir())
    assert files == ["heads.pt", "metadata.json", "temperature.json"]


def test_check_reports_shared_signature_and_dataset_license(tmp_path):
    _pack(tmp_path, "public-gpt2", encoder="openai-community/gpt2")
    _pack(tmp_path, "public-distilgpt2", encoder="distilgpt2")
    # подпись задаётся одинаковой, чтобы поймать предупреждение
    for name in ("public-gpt2", "public-distilgpt2"):
        path = tmp_path / name / "metadata.json"
        metadata = json.loads(path.read_text(encoding="utf-8"))
        metadata["questions_signature"] = "d4dcbc937478268a"
        path.write_text(json.dumps(metadata), encoding="utf-8")

    manifest = publish_hf.build_manifest(tmp_path)
    warnings = publish_hf.check_manifest(manifest)

    assert any("shared by" in warning for warning in warnings)
    assert any("license" in warning for warning in warnings)


def test_cli_check_and_missing_token(tmp_path, monkeypatch, capsys):
    packs = tmp_path / "packs"
    _pack(packs, "public-gpt2")

    assert publish_hf.main(["--check", "--packs", str(packs)]) == 0
    assert "1 packs" in capsys.readouterr().out

    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(publish_hf, "resolve_token", lambda: None)
    code = publish_hf.main(
        [
            "--publish",
            "--packs",
            str(packs),
            "--out",
            str(tmp_path / "stage"),
            "--repo",
            "owner/repo",
            "--revision",
            "v1.8.0",
        ]
    )
    assert code == 2
    assert "HF_TOKEN" in capsys.readouterr().err


def test_cli_reports_broken_pack(tmp_path, capsys):
    packs = tmp_path / "packs"
    _pack(packs, "public-gpt2")
    (packs / "public-gpt2" / "metadata.json").write_text("{}", encoding="utf-8")

    assert publish_hf.main(["--check", "--packs", str(packs)]) == 2
    assert "encoder" in capsys.readouterr().err


def test_from_stage_validates_hashes_and_requires_token(tmp_path, monkeypatch, capsys):
    packs = tmp_path / "packs"
    _pack(packs, "public-gpt2")
    manifest = publish_hf.build_manifest(packs, version="1.8.0")
    stage = tmp_path / "stage"
    publish_hf.stage(packs, stage, manifest, revision="v1.8.0")

    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(publish_hf, "resolve_token", lambda: None)
    args = ["--from-stage", str(stage), "--publish", "--repo", "owner/repo"]
    assert publish_hf.main(args) == 2
    assert "HF_TOKEN" in capsys.readouterr().err

    (stage / "packs" / "public-gpt2" / "heads.pt").write_bytes(b"tampered")
    assert publish_hf.main(args) == 2
    assert "sha256" in capsys.readouterr().err
