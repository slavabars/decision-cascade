"""Тесты A/B-замера состава признаков риск-модели (`eval/risk/features.py`)."""
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from eval.risk.features import (
    BASE_WIDTH,
    EXTRA_FEATURE_NAMES,
    aggregate_loo,
    feature_variants,
    head_features,
    main,
    mean_pairwise_cosine,
)
from src.risk import FEATURE_NAMES, decision_features
from src.sampler.questions import NoulSpec
from src.schemas import NoulResult


def _noul(p: float) -> NoulResult:
    return NoulResult(noul=p, confidence=abs(2 * p - 1))


class _StubHead:
    """Голова без весов вариантов: близость направлений равна 0."""

    linear = None


def _stub_heads(specs):
    return SimpleNamespace(heads={name: _StubHead() for name in specs})


SPECS = {"q1": NoulSpec("?q1"), "q2": NoulSpec("?q2")}
DECISIONS = {"q1": [_noul(0.9), _noul(0.2)], "q2": [_noul(0.7)]}


def _matrix(decisions: dict[str, list[NoulResult]]) -> np.ndarray:
    return np.vstack([
        decision_features(decision)
        for group in decisions.values()
        for decision in group
    ])


def test_feature_variants_cover_expected_sets():
    matrix = _matrix(DECISIONS)
    variants = feature_variants(
        _stub_heads(SPECS), SPECS, DECISIONS, DECISIONS, matrix, matrix
    )
    extra = {f"+{name}" for name in EXTRA_FEATURE_NAMES}
    assert extra == {"+log_odds_gap", "+n_classes", "+noul_logit"}
    assert set(variants) == {
        "confidence", "base(3)", "production", "+option_similarity",
    } | extra


def test_feature_variants_widths():
    matrix = _matrix(DECISIONS)
    variants = feature_variants(
        _stub_heads(SPECS), SPECS, DECISIONS, DECISIONS, matrix, matrix
    )
    widths = {name: columns.shape[1] for name, (columns, _) in variants.items()}
    assert widths["confidence"] == 1
    assert widths["base(3)"] == BASE_WIDTH
    # production — ровно то, что подключено в сервисе
    assert widths["production"] == len(FEATURE_NAMES)
    # + один новый признак к базовым
    for name in EXTRA_FEATURE_NAMES:
        assert widths[f"+{name}"] == BASE_WIDTH + 1
    # единственный вариант, требующий поля в схеме
    assert widths["+option_similarity"] == len(FEATURE_NAMES) + 1


def test_production_variant_keeps_first_rows_aligned():
    matrix = _matrix(DECISIONS)
    variants = feature_variants(
        _stub_heads(SPECS), SPECS, DECISIONS, DECISIONS, matrix, matrix
    )
    np.testing.assert_allclose(variants["production"][0], matrix)
    np.testing.assert_allclose(
        variants["base(3)"][0], matrix[:, :BASE_WIDTH]
    )


def test_mean_pairwise_cosine():
    assert mean_pairwise_cosine(torch.ones(3, 4)) == pytest.approx(1.0)
    assert mean_pairwise_cosine(torch.tensor([[1.0, 0.0], [0.0, 1.0]])) == 0.0
    assert mean_pairwise_cosine(None) == 0.0


def test_head_features_repeat_per_question():
    decisions = {"q1": [_noul(0.9)] * 3, "q2": [_noul(0.8)] * 2}
    matrix = head_features(_stub_heads(SPECS), SPECS, decisions)
    assert matrix.shape == (5, 1)  # 3 + 2 строки, порядок как в flatten
    assert (matrix == 0.0).all()


def test_cli_runs_offline_on_support_specs(tmp_path, capsys):
    rows = main([
        "--specs", "support", "--encoder", "hash", "--dim", "16",
        "--limit", "80", "--train-steps", "30", "--seeds", "0",
        "--out-dir", str(tmp_path),
    ])
    assert capsys.readouterr().out  # сводная таблица печатается
    variants = {row["variant"] for row in rows}
    assert {"confidence", "base(3)", "production", "+option_similarity"} <= variants
    for row in rows:
        assert row["n_seeds"] == 1
        assert 0.0 <= row["aurc"]
        assert 0.0 <= row["acc_at_0.7_coverage"] <= 1.0
    assert (tmp_path / "features.json").exists()


def test_aggregate_loo_macro_averages_over_questions():
    def row(question, loo, indist):
        return {
            "variant": "v", "question": question, "seed": 0,
            "loo_aurc": loo, "loo_acc_at_0.7": 0.5,
            "indist_aurc": indist, "indist_acc_at_0.7": 0.6,
        }

    aggregated = aggregate_loo([row("q1", 0.2, 0.1), row("q2", 0.4, 0.3)])
    assert len(aggregated) == 1
    entry = aggregated[0]
    assert entry["variant"] == "v"
    assert entry["n_seeds"] == 1
    # макро-среднее по вопросам, а не по строкам
    assert entry["loo_aurc"] == pytest.approx(0.3)
    assert entry["indist_aurc"] == pytest.approx(0.2)


def test_cli_loo_writes_per_question_metrics(tmp_path):
    main([
        "--specs", "support", "--encoder", "hash", "--dim", "16",
        "--limit", "200", "--train-steps", "30", "--seeds", "0", "--loo",
        "--out-dir", str(tmp_path),
    ])
    payload = json.loads((tmp_path / "features.json").read_text(encoding="utf-8"))

    rows = payload["loo_per_question"]
    questions = sorted({row["question"] for row in rows})
    variants = {row["variant"] for row in rows}
    assert len(questions) >= 2
    assert {"confidence", "base(3)", "production"} <= variants
    # на каждый вариант — ровно по строке на вопрос (1 seed)
    for variant in variants:
        assert sum(row["variant"] == variant for row in rows) == len(questions)
    for row in rows:
        # обучение идёт на всех вопросах, кроме отложенного
        assert row["n_val"] > 0 and row["n_test"] > 0
        assert 0.0 <= row["loo_aurc"] <= 1.0
        assert 0.0 <= row["loo_acc_at_0.7"] <= 1.0
    assert {entry["variant"] for entry in payload["loo"]} == variants
