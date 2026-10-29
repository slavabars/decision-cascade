"""Многодоменные паки: один сервис, несколько доменов, выбор по подписи.

Раскладка ``checkpoints/packs/<домен>/`` (``heads.pt`` + ``metadata.json``)
позволяет держать в одном процессе паки разных наборов вопросов. Пак выбирается
по ``questions_signature``, а не по имени каталога, поэтому переименование пака
ничего не ломает. Плоская раскладка ``<dir>/<signature>.pt`` остаётся рабочей.
"""
from __future__ import annotations

import json
import logging

import pytest

from api.server.service import DecisionService
from src.decision_head.heads import DecisionHeads
from src.sampler.questions import ChoiceSpec, NoulSpec, ScoreSpec, questions_signature

DIM = 8


def _support_questions():
    return {
        "urgent": NoulSpec(instructions="Is this urgent?"),
        "team": ChoiceSpec(
            instructions="Which team should handle it?",
            options=("billing", "technical", "sales"),
        ),
    }


def _routing_questions():
    return {
        "route": ChoiceSpec(
            instructions="Which model should answer?",
            options=("small", "large"),
        ),
        "benefit": ScoreSpec(
            instructions="How much does the large model help?",
            levels=("low", "medium", "high"),
        ),
    }


def _write_pack(root, name: str, questions, *, seed: int = 0, dim: int = DIM):
    """Собирает пак ``<root>/<name>/``: heads.pt + metadata.json."""
    pack = root / name
    pack.mkdir(parents=True, exist_ok=True)
    heads = DecisionHeads(dim, questions, seed=seed)
    heads.save_checkpoint(pack / "heads.pt")
    signature = questions_signature(questions)
    (pack / "metadata.json").write_text(
        json.dumps(
            {
                "schema_version": "1",
                "feature_dim": dim,
                "questions_signature": signature,
                "questions": {name: {"type": "noul"} for name in questions},
            }
        ),
        encoding="utf-8",
    )
    return pack, signature


def test_finds_pack_by_signature_in_subdirectories(tmp_path):
    support, support_sig = _write_pack(tmp_path, "support", _support_questions())
    routing, routing_sig = _write_pack(tmp_path, "routing", _routing_questions())
    assert support_sig != routing_sig

    service = DecisionService(dim=DIM, checkpoint_dir=tmp_path)
    assert service.warmup() == 2

    # Каждый набор вопросов получает свой пак, а не первый попавшийся.
    assert service._find_checkpoint(_support_questions()) == support / "heads.pt"
    assert service._find_checkpoint(_routing_questions()) == routing / "heads.pt"
    assert service.is_trained(_support_questions()) is True
    assert service.is_trained(_routing_questions()) is True


def test_decisions_come_from_the_matching_domain(tmp_path):
    _write_pack(tmp_path, "support", _support_questions())
    _write_pack(tmp_path, "routing", _routing_questions())
    service = DecisionService(dim=DIM, checkpoint_dir=tmp_path)
    service.warmup()

    routing = service.decide("Which model should answer this prompt?", _routing_questions())
    assert set(routing) == {"route", "benefit"}
    assert routing["route"].choice in {"small", "large"}

    support = service.decide("Billing is broken", _support_questions())
    assert set(support) == {"urgent", "team"}
    assert support["team"].choice in {"billing", "technical", "sales"}


def test_pack_dir_name_is_irrelevant(tmp_path):
    """Выбор идёт по подписи, поэтому имя каталога можно менять свободно."""
    _write_pack(tmp_path, "some-random-name", _support_questions())
    service = DecisionService(dim=DIM, checkpoint_dir=tmp_path)
    assert service.warmup() == 1
    assert service._find_checkpoint(_support_questions()) is not None


def test_flat_layout_still_works(tmp_path):
    questions = _support_questions()
    signature = questions_signature(questions)
    DecisionHeads(DIM, questions, seed=0).save_checkpoint(tmp_path / f"{signature}.pt")

    service = DecisionService(dim=DIM, checkpoint_dir=tmp_path)
    assert service.warmup() == 1
    assert service.is_trained(questions) is True


def test_flat_layout_wins_over_subdirectory(tmp_path):
    """Старая раскладка ``<sig>.pt`` приоритетнее пакета: это явный путь."""
    questions = _support_questions()
    signature = questions_signature(questions)
    flat = DecisionHeads(DIM, questions, seed=0).save_checkpoint(
        tmp_path / f"{signature}.pt"
    )
    _write_pack(tmp_path, "aaa-support", questions, seed=1)

    service = DecisionService(dim=DIM, checkpoint_dir=tmp_path)
    assert service._find_checkpoint(questions) == flat


def test_duplicate_signature_keeps_first_and_warns(tmp_path, caplog):
    questions = _support_questions()
    first, _ = _write_pack(tmp_path, "a-support", questions, seed=0)
    second, _ = _write_pack(tmp_path, "b-support", questions, seed=1)

    service = DecisionService(dim=DIM, checkpoint_dir=tmp_path)
    with caplog.at_level(logging.WARNING, logger="decision.service"):
        assert service.warmup() == 1
    messages = [record.message for record in caplog.records]
    assert any("already served by" in message for message in messages)
    # Первый по сортировке — источник; второй не подменяет его молча.
    assert service.info()["heads"][0]["source"] == str(first / "heads.pt")
    assert service._find_checkpoint(questions) == first / "heads.pt"
    assert str(second) not in str(service.info()["heads"][0]["source"])


def test_readiness_counts_packs_in_subdirectories(tmp_path):
    _write_pack(tmp_path, "support", _support_questions())
    _write_pack(tmp_path, "routing", _routing_questions())

    service = DecisionService(
        dim=DIM, checkpoint_dir=tmp_path, untrained_policy="refuse"
    )
    # Паки видны ещё до прогрева: readiness не должен врать про их отсутствие.
    assert service.readiness()["checkpoints"] == 2

    service.warmup()
    report = service.readiness()
    assert report["ready"] is True
    assert report["checkpoints"] == 2
    assert report["heads_loaded"] == 2


def test_info_reports_source_for_each_pack(tmp_path):
    support, _ = _write_pack(tmp_path, "support", _support_questions())
    _write_pack(tmp_path, "routing", _routing_questions())

    service = DecisionService(dim=DIM, checkpoint_dir=tmp_path)
    service.warmup()
    heads = {entry["signature"]: entry for entry in service.info()["heads"]}
    assert heads[questions_signature(_support_questions())]["source"] == str(
        support / "heads.pt"
    )
    assert all(entry["trained"] for entry in heads.values())


def test_pack_without_metadata_is_not_a_pack(tmp_path):
    """Каталог без metadata.json не считается паком — не гадаем по heads.pt."""
    orphan = tmp_path / "orphan"
    orphan.mkdir()
    DecisionHeads(DIM, _support_questions(), seed=0).save_checkpoint(
        orphan / "heads.pt"
    )

    service = DecisionService(dim=DIM, checkpoint_dir=tmp_path)
    assert service.warmup() == 0
    assert service._find_checkpoint(_support_questions()) is None


def test_unreadable_metadata_does_not_break_other_packs(tmp_path, caplog):
    _write_pack(tmp_path, "good", _support_questions())
    broken = tmp_path / "broken"
    broken.mkdir()
    DecisionHeads(DIM, _routing_questions(), seed=0).save_checkpoint(
        broken / "heads.pt"
    )
    (broken / "metadata.json").write_text("{not json", encoding="utf-8")

    service = DecisionService(dim=DIM, checkpoint_dir=tmp_path)
    with caplog.at_level(logging.WARNING, logger="decision.service"):
        service.warmup()
    assert service._find_checkpoint(_support_questions()) is not None
    assert any("cannot read pack metadata" in record.message for record in caplog.records)


@pytest.mark.parametrize("layout", ["flat", "pack"])
def test_wrong_dim_pack_is_skipped(tmp_path, layout):
    """Пак другой размерности не подменяет головы сервиса."""
    questions = _support_questions()
    signature = questions_signature(questions)
    if layout == "flat":
        path = tmp_path / f"{signature}.pt"
    else:
        pack = tmp_path / "pack"
        pack.mkdir()
        path = pack / "heads.pt"
        (pack / "metadata.json").write_text(
            json.dumps({"questions_signature": signature}), encoding="utf-8"
        )
    DecisionHeads(DIM * 2, questions, seed=0).save_checkpoint(path)

    service = DecisionService(dim=DIM, checkpoint_dir=tmp_path)
    service.warmup()
    assert service.is_trained(questions) is False


def test_pack_with_matching_dim_wins_over_alphabetical_order(tmp_path):
    """Один набор вопросов под разные энкодеры: берём совместимый по dim пак."""
    questions = _support_questions()
    _write_pack(tmp_path, "a-big", questions, dim=DIM * 2)
    right, _ = _write_pack(tmp_path, "b-right", questions, dim=DIM)

    service = DecisionService(dim=DIM, checkpoint_dir=tmp_path)
    assert service._find_checkpoint(questions, DIM) == right / "heads.pt"
    # без подсказки dim — первый по сортировке, как и раньше
    assert service._find_checkpoint(questions) is not None
    service.warmup()
    assert service.is_trained(questions) is True
    assert service.info()["heads"][0]["source"] == str(right / "heads.pt")
