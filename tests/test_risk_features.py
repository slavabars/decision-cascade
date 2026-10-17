"""Тесты A/B-замера состава признаков риск-модели (`eval/risk/features.py`)."""
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from eval.risk.core import GATE_AB_METRICS, GATE_LOO_METRICS
from eval.risk.features import (
    BASE_WIDTH,
    EXTRA_FEATURE_NAMES,
    _Prepared,
    aggregate_gate,
    aggregate_loo,
    aggregate_shots,
    calibration_report,
    feature_variants,
    gate_vs_baseline,
    head_features,
    main,
    mean_pairwise_cosine,
    question_onehot,
    risk_report,
    run_loo,
    shots_report,
    with_state_variants,
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
        "confidence", "base(3)", "random", "production", "+option_similarity",
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


def test_question_onehot_rows_follow_question_major_order():
    decisions = {"q1": [_noul(0.9)] * 3, "q2": [_noul(0.8)] * 2}
    onehot = question_onehot(SPECS, decisions)
    assert onehot.shape == (5, 2)
    np.testing.assert_allclose(onehot, [[1, 0], [1, 0], [1, 0], [0, 1], [0, 1]])


def test_question_variants_add_one_hot_intercept():
    matrix = _matrix(DECISIONS)
    variants = feature_variants(
        _stub_heads(SPECS), SPECS, DECISIONS, DECISIONS, matrix, matrix,
        question=True,
    )
    assert variants["prod+question"][0].shape == (3, len(FEATURE_NAMES) + 2)
    # n_classes (колонка 4) выброшен, вместо него — по смещению на вопрос
    assert variants["no_n_classes+question"][0].shape == (3, len(FEATURE_NAMES) - 1 + 2)
    np.testing.assert_allclose(
        variants["prod+question"][0][:, : len(FEATURE_NAMES)], matrix
    )
    assert "prod+question" not in feature_variants(
        _stub_heads(SPECS), SPECS, DECISIONS, DECISIONS, matrix, matrix
    )


def test_cli_question_adds_intercept_variants(tmp_path):
    main([
        "--specs", "support", "--encoder", "hash", "--dim", "16",
        "--limit", "120", "--train-steps", "30", "--seeds", "0",
        "--question", "--loo", "--out-dir", str(tmp_path),
    ])
    payload = json.loads((tmp_path / "features.json").read_text(encoding="utf-8"))

    variants = set(payload["per_seed"])
    assert {"prod+question", "no_n_classes+question"} <= variants
    assert {"prod+question", "no_n_classes+question"} <= {
        row["variant"] for row in payload["loo"]
    }


def test_cli_runs_offline_on_support_specs(tmp_path, capsys):
    rows = main([
        "--specs", "support", "--encoder", "hash", "--dim", "16",
        "--limit", "80", "--train-steps", "30", "--seeds", "0",
        "--out-dir", str(tmp_path),
    ])
    assert capsys.readouterr().out  # сводная таблица печатается
    variants = {row["variant"] for row in rows}
    assert {"confidence", "base(3)", "production", "+option_similarity"} <= variants
    assert "random" in variants
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
            "loo_ece": 0.10, "loo_brier": 0.20,
            "indist_aurc": indist, "indist_acc_at_0.7": 0.6,
            "indist_ece": 0.08, "indist_brier": 0.18,
        }

    aggregated = aggregate_loo([row("q1", 0.2, 0.1), row("q2", 0.4, 0.3)])
    assert len(aggregated) == 1
    entry = aggregated[0]
    assert entry["variant"] == "v"
    assert entry["n_seeds"] == 1
    # макро-среднее по вопросам, а не по строкам
    assert entry["loo_aurc"] == pytest.approx(0.3)
    assert entry["indist_aurc"] == pytest.approx(0.2)
    # калибровка тоже агрегируется: LOO против in-dist
    assert entry["loo_ece"] == pytest.approx(0.10)
    assert entry["indist_ece"] == pytest.approx(0.08)
    assert entry["loo_brier_std"] == pytest.approx(0.0)


def _fixture_block(questions, rows_per_question, rng):
    """Один вопрос = по одному «своему» признаку, поэтому вопросы расходятся."""
    matrix, correct, groups = [], [], []
    for question in questions:
        block = rng.uniform(-1.0, 1.0, size=(rows_per_question, len(FEATURE_NAMES)))
        important = 0 if question == "q1" else 1
        score = block[:, important] + 0.4 * rng.standard_normal(rows_per_question)
        matrix.append(block)
        correct.append(score > 0)
        groups.extend([question] * rows_per_question)
    return np.vstack(matrix), np.concatenate(correct), groups


def _prepared_fixture() -> _Prepared:
    """Валидация и тест по трём вопросам в порядке question-major."""
    rng = np.random.default_rng(7)
    questions = ["q1", "q2", "q3"]
    val = _fixture_block(questions, 10, rng)
    test = _fixture_block(questions, 7, rng)
    return _Prepared(
        variants={"production": (val[0], test[0])},
        correct=(val[1], test[1]),
        groups=(val[2], test[2]),
        questions=questions,
    )


def test_shots_report_endpoints_match_loo():
    """n = 0 — это ровно LOO, `shots=None` — ровно in-dist из `run_loo`."""
    prepared = _prepared_fixture()
    rows = shots_report(prepared, [0], seed=0, C=1.0)
    loo = run_loo(prepared, seed=0, C=1.0)
    assert {row["shots"] for row in rows} == {None, 0}

    for held_out in loo:
        own = [
            row for row in rows
            if row["question"] == held_out["question"]
        ]
        points = {row["shots"]: row for row in own}
        assert points[0]["aurc"] == pytest.approx(held_out["loo_aurc"])
        assert points[0]["acc_at_0.7_coverage"] == pytest.approx(
            held_out["loo_acc_at_0.7"]
        )
        assert points[None]["aurc"] == pytest.approx(held_out["indist_aurc"])


def test_shots_report_takes_nested_subsets_of_validation_only():
    prepared = _prepared_fixture()
    rows = shots_report(prepared, [0, 3, 8], seed=0)

    for question in prepared.questions:
        points = [row for row in rows if row["question"] == question]
        # всегда есть точка «все свои строки» (верхняя граница)
        assert [row["shots"] for row in points] == [None, 0, 3, 8]
        assert [row["n_own"] for row in points] == [10, 0, 3, 8]
        for row in points:
            # чужие вопросы плюс свои: тест не подмешивается
            assert row["n_train"] == 20 + row["n_own"]
            assert row["n_test"] == 7
            assert 0.0 <= row["aurc"] <= 1.0


def test_aggregate_shots_macro_averages_over_questions():
    def row(shots, question, aurc):
        return {
            "variant": "v", "question": question, "seed": 0, "shots": shots,
            "n_own": 4, "n_train": 24, "n_test": 7,
            "aurc": aurc, "eaurc": aurc / 2,
            "coverage_at_5pct_risk": 0.4, "acc_at_0.7_coverage": 0.5,
            "ece": 0.1, "brier": 0.2,
        }

    rows = [
        row(0, "q1", 0.2), row(0, "q2", 0.4),
        row(None, "q1", 0.1), row(None, "q2", 0.3),
    ]
    by_shots = {entry["shots"]: entry for entry in aggregate_shots(rows)}
    assert set(by_shots) == {0, None}
    # макро-среднее по вопросам, а не по строкам
    assert by_shots[0]["aurc"] == pytest.approx(0.3)
    assert by_shots[None]["aurc"] == pytest.approx(0.2)


def _shots_row(question, aurc, acc, *, variant, shots):
    return {
        "variant": variant, "question": question, "seed": 0, "shots": shots,
        "n_own": 0 if shots is None else shots, "n_train": 24, "n_test": 7,
        "aurc": aurc, "eaurc": aurc / 2,
        "coverage_at_5pct_risk": 0.4, "acc_at_0.7_coverage": acc,
        "ece": 0.1, "brier": 0.2,
    }


def test_gate_on_shots_curve_joins_baseline_at_same_shots():
    """Гейт по кривой: точка сравнивается с baseline при том же числе примеров."""
    rows = []
    for question in ("q1", "q2"):
        rows.append(_shots_row(question, 0.30, 0.60, variant="confidence", shots=0))
        rows.append(_shots_row(question, 0.40, 0.55, variant="production", shots=0))
        rows.append(_shots_row(question, 0.30, 0.60, variant="confidence", shots=5))
        rows.append(_shots_row(question, 0.25, 0.65, variant="production", shots=5))

    verdicts = gate_vs_baseline(rows, GATE_AB_METRICS)
    assert {v["shots"] for v in verdicts} == {0, 5}
    assert all(v["variant"] == "production" for v in verdicts)

    by_point = {(e["variant"], e["shots"]): e for e in aggregate_gate(verdicts)}
    assert by_point[("production", 0)]["accepted_rate"] == 0.0
    assert by_point[("production", 5)]["accepted_rate"] == 1.0
    assert by_point[("production", 0)]["failures"][0]["question"] == "q1"
    # смешивать точки с разным числом примеров нельзя
    assert by_point[("production", 0)]["n"] == 2


def test_cli_shots_writes_curve(tmp_path, capsys):
    main([
        "--specs", "support", "--encoder", "hash", "--dim", "16",
        "--limit", "120", "--train-steps", "30", "--seeds", "0",
        "--shots", "0", "5", "--out-dir", str(tmp_path),
    ])
    out = capsys.readouterr().out
    assert "сколько строк нового вопроса нужно" in out
    assert "все (in-dist)" in out

    payload = json.loads((tmp_path / "features.json").read_text(encoding="utf-8"))
    assert {entry["shots"] for entry in payload["shots"]} == {None, 0, 5}
    assert payload["shots_per_question"]
    for row in payload["shots_per_question"]:
        assert row["n_test"] > 0 and row["n_own"] <= row["n_train"]
        assert 0.0 <= row["aurc"] <= 1.0
        assert 0.0 <= row["ece"] <= 1.0 and 0.0 <= row["brier"] <= 1.0
    # гейт считается и по кривой: production против confidence при том же n
    gate_shots = payload["gate"]["shots"]
    assert gate_shots and {entry["variant"] for entry in gate_shots} == {"production"}
    assert {entry["shots"] for entry in gate_shots} == {None, 0, 5}
    assert "hard gate, кривая few-shot" in out


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
        # калибровка P(ошибка) — и на новом вопросе, и внутри вопроса
        assert 0.0 <= row["loo_ece"] <= 1.0
        assert 0.0 <= row["loo_brier"] <= 1.0
        assert 0.0 <= row["indist_ece"] <= 1.0
        assert 0.0 <= row["indist_brier"] <= 1.0
    assert {entry["variant"] for entry in payload["loo"]} == variants
    # агрегат LOO несёт калибровку и разброс по seed'ам
    for entry in payload["loo"]:
        for key in ("loo_ece", "loo_brier", "indist_ece", "indist_brier"):
            assert key in entry and f"{key}_std" in entry


def test_calibration_report_scores_error_probability():
    """ECE/Brier считаются на P(ошибка): идеальные баллы дают нули."""
    scores = np.array([0.9, 0.9, 0.1, 0.1])
    correct = np.array([1.0, 1.0, 0.0, 0.0])
    report = calibration_report(scores, correct)
    assert report["brier"] == pytest.approx(0.01)
    assert report["ece"] == pytest.approx(0.1)

    # баллы-константы: ECE ловит недокалиброванность, Brier — квадрат промаха
    flat = calibration_report(np.full(4, 0.5), np.ones(4))
    assert flat["ece"] == pytest.approx(0.5)
    assert flat["brier"] == pytest.approx(0.25)


def test_risk_report_includes_calibration_and_metrics():
    scores = np.array([0.9, 0.8, 0.3, 0.2])
    correct = np.array([1.0, 1.0, 0.0, 0.0])
    report = risk_report(scores, correct)
    assert {"aurc", "eaurc", "coverage_at_5pct_risk", "acc_at_0.7_coverage",
            "ece", "brier"} <= set(report)
    assert 0.0 <= report["ece"] <= 1.0
    assert 0.0 <= report["brier"] <= 1.0


def test_with_state_variants_shapes_and_standardization():
    features = np.zeros((4, 6))
    variants = {"production": (features, features)}
    states = tuple(np.arange(12, dtype=float).reshape(4, 3) for _ in range(2))

    out = with_state_variants(variants, states, [0.05, 1.0])

    assert set(out) == {
        "production", "state[0.05]", "prod+state[0.05]", "state[1]", "prod+state[1]"
    }
    assert out["production"][0] is features
    assert out["state[0.05]"][0].shape == (4, 3)
    assert out["prod+state[0.05]"][0].shape == (4, 9)
    # состояние стандартизовано по статистике валидации, значит по нулю
    assert out["state[1]"][0].mean(axis=0) == pytest.approx(np.zeros(3))


def test_cli_state_adds_state_variants(tmp_path):
    main([
        "--specs", "support", "--encoder", "hash", "--dim", "16",
        "--limit", "120", "--train-steps", "30", "--seeds", "0",
        "--state", "--state-C", "0.05", "1.0", "--out-dir", str(tmp_path),
    ])
    payload = json.loads((tmp_path / "features.json").read_text(encoding="utf-8"))

    assert payload["state_cs"] == [0.05, 1.0]
    variants = set(payload["per_seed"])
    assert {"state[0.05]", "prod+state[0.05]", "state[1]", "prod+state[1]"} <= variants
    for name in ("state[0.05]", "prod+state[1]"):
        for row in payload["per_seed"][name]:
            assert 0.0 <= row["aurc"] <= 1.0


def _loo_row(question, aurc, acc, *, variant="production", seed=0, conf=False):
    return {
        "variant": "confidence" if conf else variant,
        "question": question,
        "seed": seed,
        "n_val": 40,
        "n_test": 20,
        "loo_aurc": aurc,
        "loo_eaurc": aurc / 2,
        "loo_coverage_at_5pct_risk": 0.5,
        "loo_acc_at_0.7": acc,
        "indist_aurc": aurc,
        "indist_acc_at_0.7": acc,
    }


def test_gate_vs_baseline_flags_regression_per_question():
    rows = [
        _loo_row("q1", aurc=0.30, acc=0.80, conf=True),
        _loo_row("q1", aurc=0.28, acc=0.82),
        _loo_row("q2", aurc=0.30, acc=0.80, conf=True),
        _loo_row("q2", aurc=0.40, acc=0.75),
    ]
    verdicts = gate_vs_baseline(rows, GATE_LOO_METRICS)
    by_question = {v["question"]: v for v in verdicts}
    assert by_question["q1"]["accepted"]
    assert not by_question["q2"]["accepted"]
    assert by_question["q2"]["worse"] and by_question["q2"]["worse"][0].startswith(
        "loo_aurc"
    )
    # baseline сам с собой не сравнивается
    assert all(v["variant"] == "production" for v in verdicts)


def test_aggregate_gate_reports_pass_rate():
    rows = [
        _loo_row("q1", aurc=0.30, acc=0.80, conf=True),
        _loo_row("q1", aurc=0.28, acc=0.82),
        _loo_row("q2", aurc=0.30, acc=0.80, conf=True),
        _loo_row("q2", aurc=0.40, acc=0.75),
    ]
    aggregated = aggregate_gate(gate_vs_baseline(rows, GATE_LOO_METRICS))
    assert len(aggregated) == 1
    entry = aggregated[0]
    assert entry["variant"] == "production"
    assert entry["n"] == 2 and entry["accepted"] == 1
    assert entry["accepted_rate"] == pytest.approx(0.5)
    assert entry["failures"][0]["question"] == "q2"


def test_cli_loo_prints_gate_and_saves_it(tmp_path, capsys):
    main([
        "--specs", "support", "--encoder", "hash", "--dim", "16",
        "--limit", "200", "--train-steps", "30", "--seeds", "0", "--loo",
        "--out-dir", str(tmp_path),
    ])
    out = capsys.readouterr().out
    assert "hard gate, на новом вопросе (LOO)" in out

    payload = json.loads((tmp_path / "features.json").read_text(encoding="utf-8"))
    gate = payload["gate"]
    assert gate["baseline"] == "confidence"
    assert gate["tol"] == pytest.approx(0.0005)
    assert gate["in_dist"] and gate["loo"]
    assert {entry["variant"] for entry in gate["loo"]} >= {"production"}
    for verdict in gate["verdicts_loo"]:
        assert isinstance(verdict["accepted"], bool)
