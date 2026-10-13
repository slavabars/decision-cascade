"""Тесты версионирования схемы и миграции чекпоинтов (F4)."""
import pytest
import torch

from src.versioning import (
    CHECKPOINT_VERSION,
    is_compatible,
    migrate_checkpoint,
    parse_version,
    version_major,
)


def test_parse_version_accepts_major_and_minor():
    assert parse_version("1") == (1, 0)
    assert parse_version("2.7") == (2, 7)


@pytest.mark.parametrize("bad", ["", "abc", "1.2.3", "1.x", "-1"])
def test_parse_version_rejects_malformed(bad):
    with pytest.raises(ValueError):
        parse_version(bad)


def test_version_major():
    assert version_major("1.9") == 1


def test_is_compatible_same_major():
    assert is_compatible("1", "1.4") is True
    assert is_compatible("1.4", "1") is True


def test_is_compatible_different_major():
    assert is_compatible("1", "2") is False
    assert is_compatible("2", "1.9") is False


def test_is_compatible_none_is_legacy():
    assert is_compatible("1", None) is True


def test_is_compatible_malformed_is_false():
    assert is_compatible("1", "abc") is False


def test_migrate_checkpoint_adds_version_to_legacy():
    payload = {"dim": 4, "questions": {}, "state_dict": {}}
    migrated = migrate_checkpoint(payload)
    assert migrated["schema_version"] == CHECKPOINT_VERSION
    # исходный словарь не мутируется
    assert "schema_version" not in payload


def test_migrate_checkpoint_accepts_same_major():
    payload = {"dim": 4, "schema_version": "1.3"}
    assert migrate_checkpoint(payload)["schema_version"] == "1.3"


def test_migrate_checkpoint_rejects_future_major():
    with pytest.raises(ValueError, match="schema_version"):
        migrate_checkpoint({"dim": 4, "schema_version": "2"})


def test_checkpoint_roundtrip_records_version(tmp_path):
    from src.decision_head.heads import DecisionHeads
    from src.sampler.questions import NoulSpec, ScoreSpec

    heads = DecisionHeads(
        4, {"u": NoulSpec("urgent?"), "s": ScoreSpec("heat", ("a", "b"))}, seed=0
    )
    path = heads.save_checkpoint(tmp_path / "heads.pt")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    assert payload["schema_version"] == CHECKPOINT_VERSION

    restored = DecisionHeads.load_checkpoint(path)
    assert restored.dim == 4
    assert set(restored.questions) == {"u", "s"}


def test_checkpoint_loads_legacy_payload_without_version(tmp_path):
    from src.decision_head.heads import DecisionHeads
    from src.sampler.questions import NoulSpec, spec_to_dict

    heads = DecisionHeads(4, {"u": NoulSpec("urgent?")}, seed=0)
    legacy = {
        "dim": 4,
        "questions": {"u": spec_to_dict(NoulSpec("urgent?"))},
        "state_dict": heads.state_dict(),
    }
    path = tmp_path / "legacy.pt"
    torch.save(legacy, path)

    restored = DecisionHeads.load_checkpoint(path)
    assert restored.dim == 4
    assert set(restored.questions) == {"u"}


def test_checkpoint_loads_version_rejected_for_future_major(tmp_path):
    from src.decision_head.heads import DecisionHeads

    heads = DecisionHeads(4, {}, seed=0)
    future = {
        "schema_version": "2",
        "dim": 4,
        "questions": {},
        "state_dict": heads.state_dict(),
    }
    path = tmp_path / "future.pt"
    torch.save(future, path)
    with pytest.raises(ValueError, match="schema_version"):
        DecisionHeads.load_checkpoint(path)
