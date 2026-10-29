"""Обратная совместимость паков: артефакт 0.8.0 читается текущей версией.

`tests/fixtures/pack_v0.8.0.pt` собран кодом тега `v0.8.0` (dim=8, три вопроса
всех типов) и лежит в репозитории как эталон. Если текущая версия перестанет
читать старый пак, тест упадёт — это и есть обещание из `docs/stability.md` §1.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import torch

from api.server.service import DecisionService
from src.decision_head.heads import DecisionHeads
from src.sampler.questions import ChoiceSpec, NoulSpec, ScoreSpec, questions_signature
from src.versioning import CHECKPOINT_VERSION, is_compatible, migrate_checkpoint

FIXTURE = Path(__file__).parent / "fixtures" / "pack_v0.8.0.pt"
DIM = 8


def _questions() -> dict[str, NoulSpec | ChoiceSpec | ScoreSpec]:
    """Ровно те вопросы, под которые обучен эталонный пак 0.8.0."""
    return {
        "urgent": NoulSpec(instructions="Is this urgent?"),
        "team": ChoiceSpec(
            instructions="Which team should handle it?",
            options=("billing", "technical", "sales"),
        ),
        "severity": ScoreSpec(
            instructions="How severe is it?",
            levels=("low", "medium", "high"),
        ),
    }


def test_fixture_is_small_and_committed():
    assert FIXTURE.is_file(), "эталонный пак 0.8.0 пропал из tests/fixtures"
    assert FIXTURE.stat().st_size < 64 * 1024


def test_pack_from_0_8_loads_with_current_code():
    heads = DecisionHeads.load_checkpoint(FIXTURE)
    assert heads.dim == DIM
    assert heads.choice_head == "linear"
    assert set(heads.questions) == {"urgent", "team", "severity"}
    assert heads.temperatures() == {"urgent": 1.0, "team": 1.0, "severity": 1.0}


def test_pack_0_8_payload_migrates_to_current_schema():
    payload = torch.load(FIXTURE, map_location="cpu", weights_only=True)
    assert is_compatible(CHECKPOINT_VERSION, payload.get("schema_version")) is True
    migrated = migrate_checkpoint(payload)
    assert migrated["schema_version"] == CHECKPOINT_VERSION
    assert migrated["dim"] == DIM


def test_service_serves_decisions_from_0_8_pack(tmp_path):
    """Пак 0.8.0, положенный под текущую подпись, даёт обученные решения."""
    questions = _questions()
    shutil.copy(FIXTURE, tmp_path / f"{questions_signature(questions)}.pt")
    service = DecisionService(dim=DIM, checkpoint_dir=tmp_path)
    assert service.warmup() == 1
    assert service.is_trained(questions) is True

    result = service.decide("Server is down and billing is broken", questions)
    assert set(result) == {"urgent", "team", "severity"}
    assert result["team"].choice in {"billing", "technical", "sales"}
    assert 0.0 <= result["severity"].score <= 2.0
    assert set(result["severity"].probabilities) == {"low", "medium", "high"}
    assert 0.0 <= result["urgent"].noul <= 1.0


def test_current_pack_roundtrips(tmp_path):
    """Обратная сторона обещания: пак, собранный сейчас, читается этим же кодом."""
    questions = _questions()
    path = DecisionHeads(DIM, questions, seed=0).save_checkpoint(tmp_path / "heads.pt")
    loaded = DecisionHeads.load_checkpoint(path)
    assert loaded.dim == DIM
    assert loaded.temperatures() == {"urgent": 1.0, "team": 1.0, "severity": 1.0}


@pytest.mark.parametrize("version", ["1", "1.9"])
def test_current_code_reads_1x_packs(version):
    """Линейка 1.x обещана совместимой по `schema_version`."""
    assert is_compatible(CHECKPOINT_VERSION, version) is True
