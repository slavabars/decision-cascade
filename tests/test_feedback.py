"""Дообучение на обратной связи: replay, гейт «не навреди», запись пака."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from api.server.service import HashStateEncoder
from eval.benchmark.core import evaluate_benchmark
from src.decision_head.heads import DecisionHeads
from src.sampler.questions import ChoiceSpec, NoulSpec, questions_signature
from training.feedback import run

DIM = 64
SPECS = {
    "urgent": NoulSpec(instructions="Is this urgent?"),
    "team": ChoiceSpec(instructions="Which team?", options=("billing", "tech")),
}

#: Старый домен: слова alpha/beta -> (urgent, team).
OLD = {
    "alpha": {"urgent": 0, "team": "billing"},
    "beta": {"urgent": 1, "team": "tech"},
}
#: Новый домен: другие слова и обратная связь меток (дрейф).
NEW = {
    "gamma": {"urgent": 1, "team": "tech"},
    "delta": {"urgent": 0, "team": "billing"},
}


def _examples(domain: dict[str, dict], n: int, prefix: str) -> list[dict]:
    words = list(domain)
    return [
        {
            "id": f"{prefix}-{index}",
            "state": f"{words[index % len(words)]} request number {index}",
            "labels": domain[words[index % len(words)]],
        }
        for index in range(n)
    ]


def _write_pack(tmp_path, examples: list[dict]) -> Path:
    pack = tmp_path / "pack"
    pack.mkdir()
    heads = DecisionHeads(DIM, SPECS, seed=0)
    # базовая голова должна быть обучена, иначе сравнение бессмысленно
    from eval.benchmark.core import build_targets, encode_examples
    from training.sft.train import SFTTrainer

    encoder = HashStateEncoder(DIM)
    features = encode_examples(SPECS, examples, encoder)
    targets = {
        name: build_targets(spec, [ex["labels"][name] for ex in examples])
        for name, spec in SPECS.items()
    }
    SFTTrainer(heads, lr=0.1).fit(features, targets, steps=200)
    heads.save_checkpoint(pack / "heads.pt")
    (pack / "metadata.json").write_text(
        json.dumps(
            {
                "schema_version": "1",
                "encoder": "hash",
                "feature_dim": DIM,
                "questions_signature": questions_signature(SPECS),
            }
        ),
        encoding="utf-8",
    )
    (pack / "dataset.jsonl").write_text(
        "\n".join(json.dumps(example) for example in examples) + "\n",
        encoding="utf-8",
    )
    return pack


def _write_jsonl(path, examples: list[dict]):
    path.write_text(
        "\n".join(json.dumps(example) for example in examples) + "\n",
        encoding="utf-8",
    )
    return path


# --- чтение и валидация ------------------------------------------------------


def test_read_examples_accepts_valid_rows(tmp_path):
    path = _write_jsonl(tmp_path / "f.jsonl", _examples(OLD, 3, "old"))
    examples = run.read_examples(path, SPECS)
    assert len(examples) == 3
    assert examples[0]["labels"] == {"urgent": 0, "team": "billing"}


def test_read_examples_rejects_choice_outside_options(tmp_path):
    path = tmp_path / "f.jsonl"
    path.write_text(
        json.dumps({"state": "s", "labels": {"urgent": 1, "team": "sales"}}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="not one of"):
        run.read_examples(path, SPECS)


def test_read_examples_rejects_missing_and_unknown_labels(tmp_path):
    path = tmp_path / "f.jsonl"
    path.write_text(
        json.dumps({"state": "s", "labels": {"urgent": 1}}) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="no labels for questions"):
        run.read_examples(path, SPECS)

    path.write_text(
        json.dumps({"state": "s", "labels": {"urgent": 1, "team": "tech", "x": 1}})
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unknown questions"):
        run.read_examples(path, SPECS)


def test_read_examples_rejects_empty_file_and_bad_state(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_text("\n\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no examples"):
        run.read_examples(path, SPECS)

    path.write_text(
        json.dumps({"state": "  ", "labels": {"urgent": 1, "team": "tech"}}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="no non-empty 'state'"):
        run.read_examples(path, SPECS)


def test_validate_label_types():
    from src.sampler.questions import ScoreSpec

    score = ScoreSpec(instructions="how?", levels=("low", "high"))
    assert run.validate_label(score, 1, "q") == 1
    with pytest.raises(TypeError, match="must be an int"):
        run.validate_label(score, "1", "q")
    with pytest.raises(ValueError, match="out of range"):
        run.validate_label(score, 2, "q")
    with pytest.raises(ValueError, match="must be 0/1"):
        run.validate_label(SPECS["urgent"], 0.5, "urgent")


# --- разбиения ---------------------------------------------------------------


def test_split_holdout_sizes_and_errors():
    examples = _examples(OLD, 10, "x")
    train, holdout = run.split_holdout(examples, 0.2, seed=0)
    assert len(train) == 8
    assert len(holdout) == 2
    assert not {e["id"] for e in train} & {e["id"] for e in holdout}
    with pytest.raises(ValueError, match="too small"):
        run.split_holdout(examples[:3], 0.2)
    with pytest.raises(ValueError, match="holdout fraction"):
        run.split_holdout(examples, 0.0)


def test_sample_replay_fraction():
    examples = _examples(OLD, 10, "x")
    assert run.sample_replay(examples, 0.0) == []
    assert len(run.sample_replay(examples, 1.0)) == 10
    assert len(run.sample_replay(examples, 0.5)) == 5
    assert run.sample_replay([], 0.5) == []
    # доля > 1 повторяет старые строки: иначе при большом K новый домен топит старый
    doubled = run.sample_replay(examples, 2.0)
    assert len(doubled) == 20
    assert len({ex["id"] for ex in doubled}) == 10
    with pytest.raises(ValueError, match="replay fraction"):
        run.sample_replay(examples, -0.5)


def test_split_retention_keeps_everything_when_small():
    examples = _examples(OLD, 4, "x")
    train, holdout = run.split_retention(examples, 0.2)
    assert train == examples
    assert holdout == []


# --- гейт --------------------------------------------------------------------


def test_gate_accepts_improvement_without_old_damage():
    verdict = run.gate(
        before_new={"accuracy": 0.5, "ece": 0.2},
        after_new={"accuracy": 0.7, "ece": 0.1},
        before_old={"accuracy": 0.9, "ece": 0.05},
        after_old={"accuracy": 0.9, "ece": 0.06},
    )
    assert verdict["accepted"] is True
    assert verdict["new_gain"] == pytest.approx(0.2)
    assert verdict["old_drop"] == pytest.approx(0.0)
    assert verdict["reasons"] == []


def test_gate_rejects_without_gain():
    verdict = run.gate(
        before_new={"accuracy": 0.6, "ece": 0.1},
        after_new={"accuracy": 0.6, "ece": 0.1},
        before_old=None,
        after_old=None,
        min_new_gain=0.01,
    )
    assert verdict["accepted"] is False
    assert "new domain gain" in verdict["reasons"][0]
    assert verdict["old_drop"] is None


def test_gate_rejects_catastrophic_forgetting():
    verdict = run.gate(
        before_new={"accuracy": 0.5, "ece": 0.2},
        after_new={"accuracy": 0.9, "ece": 0.05},
        before_old={"accuracy": 0.9, "ece": 0.05},
        after_old={"accuracy": 0.5, "ece": 0.2},
    )
    assert verdict["accepted"] is False
    assert "old domain drop" in verdict["reasons"][0]


# --- дообучение и CLI --------------------------------------------------------


def test_retrain_recovers_new_domain_with_replay(tmp_path):
    old = _examples(OLD, 60, "old")
    new = _examples(NEW, 60, "new")
    encoder = HashStateEncoder(DIM)
    heads, _ = run.load_pack(_write_pack(tmp_path, old))

    before = run.summarize(evaluate_benchmark(SPECS, new, encoder, heads))
    trained = run.retrain(
        heads, SPECS, encoder, new[:40], old[:30], steps=300, lr=0.1
    )
    after = run.summarize(evaluate_benchmark(SPECS, new[40:], encoder, trained))
    assert after["accuracy"] > before["accuracy"]


def test_cli_writes_pack_and_metadata(tmp_path):
    old = _examples(OLD, 200, "old")
    new = _examples(NEW, 120, "new")
    pack = _write_pack(tmp_path, old)
    feedback = _write_jsonl(tmp_path / "feedback.jsonl", new)
    out = tmp_path / "pack-v2"

    report = run.main(
        [
            "--pack",
            str(pack),
            "--feedback",
            str(feedback),
            "--out",
            str(out),
            "--replay",
            "1.0",
            "--steps",
            "300",
            "--lr",
            "0.1",
            "--holdout",
            "0.25",
        ],
        encoder=HashStateEncoder(DIM),
    )
    assert report["gate"]["accepted"] is True
    assert report["written"] is True
    assert (out / "heads.pt").is_file()
    metadata = json.loads((out / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["feedback"]["n"] == 120
    assert metadata["feedback"]["gate"]["accepted"] is True
    assert metadata["feedback"]["temperature_recalibrated"] is False
    merged = (out / "dataset.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(merged) == len(old) + report["n_train"]
    # пак читается сервисом
    reloaded = DecisionHeads.load_checkpoint(out / "heads.pt")
    assert set(reloaded.questions) == set(SPECS)


def test_cli_rejects_when_gain_is_impossible(tmp_path):
    old = _examples(OLD, 120, "old")
    new = _examples(NEW, 60, "new")
    pack = _write_pack(tmp_path, old)
    feedback = _write_jsonl(tmp_path / "feedback.jsonl", new)
    out = tmp_path / "pack-v2"

    report = run.main(
        [
            "--pack",
            str(pack),
            "--feedback",
            str(feedback),
            "--out",
            str(out),
            "--steps",
            "50",
            "--min-new-gain",
            "0.99",
            "--max-old-drop",
            "0.0",
        ],
        encoder=HashStateEncoder(DIM),
    )
    assert report["gate"]["accepted"] is False
    assert report["written"] is False
    assert not (out / "heads.pt").exists()
    # отчёт всё равно записан — чтобы было видно, почему отказали
    saved = json.loads((out / "feedback.report.json").read_text(encoding="utf-8"))
    assert saved["gate"]["accepted"] is False


def test_cli_dry_run_does_not_write_pack(tmp_path):
    old = _examples(OLD, 120, "old")
    new = _examples(NEW, 60, "new")
    pack = _write_pack(tmp_path, old)
    feedback = _write_jsonl(tmp_path / "feedback.jsonl", new)
    out = tmp_path / "pack-v2"

    report = run.main(
        [
            "--pack",
            str(pack),
            "--feedback",
            str(feedback),
            "--out",
            str(out),
            "--steps",
            "50",
            "--dry-run",
        ],
        encoder=HashStateEncoder(DIM),
    )
    assert report["written"] is False
    assert not (out / "heads.pt").exists()


def test_load_pack_errors(tmp_path):
    with pytest.raises(ValueError, match="no heads.pt"):
        run.load_pack(tmp_path)


def test_cli_control_reports_net_gain(tmp_path):
    old = _examples(OLD, 200, "old")
    new = _examples(NEW, 120, "new")
    pack = _write_pack(tmp_path, old)
    feedback = _write_jsonl(tmp_path / "feedback.jsonl", new)
    out = tmp_path / "pack-v2"

    report = run.main(
        [
            "--pack",
            str(pack),
            "--feedback",
            str(feedback),
            "--out",
            str(out),
            "--steps",
            "100",
            "--lr",
            "0.1",
            "--control",
            "--dry-run",
        ],
        encoder=HashStateEncoder(DIM),
    )
    assert report["control"] is not None
    assert "new" in report["control"] and "old" in report["control"]
    assert report["net"] is not None
    assert set(report["net"]) == {"new_gain", "old_delta"}
