"""Тесты conformal-сертификата: граница Клоппера–Пирсона, порог, файл."""
from __future__ import annotations

import json

import numpy as np
import pytest

from src.conformal import (
    METHOD,
    clopper_pearson_upper,
    load_certificate,
    save_certificate,
    selective_certificate,
    validate_certificate,
)


def _dataset(n_good=90, n_bad=10):
    scores = np.arange(1, n_good + n_bad + 1, dtype=float) / (n_good + n_bad)
    correct = np.asarray([1.0] * n_good + [0.0] * n_bad)
    return scores, correct


def test_clopper_pearson_upper_is_conservative():
    # ноль ошибок на 59 примерах уже даёт границу ниже 5%
    assert clopper_pearson_upper(0, 59, 0.05) <= 0.05
    assert clopper_pearson_upper(0, 10, 0.05) > 0.05
    # граница растёт с числом ошибок и падает с размером выборки
    assert clopper_pearson_upper(1, 100, 0.05) > clopper_pearson_upper(0, 100, 0.05)
    assert clopper_pearson_upper(0, 1000, 0.05) < clopper_pearson_upper(0, 100, 0.05)
    # крайние случаи
    assert clopper_pearson_upper(0, 0, 0.05) == 1.0
    assert clopper_pearson_upper(5, 5, 0.05) == 1.0
    with pytest.raises(ValueError):
        clopper_pearson_upper(6, 5, 0.05)
    with pytest.raises(ValueError):
        clopper_pearson_upper(0, 10, 0.0)


def test_certificate_picks_widest_threshold_within_alpha():
    scores, correct = _dataset()
    certificate = selective_certificate(
        scores, correct, alpha=0.1, delta=0.05, min_coverage=0.0
    )

    assert certificate["method"] == METHOD
    assert certificate["upper_bound"] <= certificate["alpha"]
    assert certificate["coverage"] >= 0.9
    assert certificate["empirical_risk"] <= certificate["alpha"]
    assert certificate["threshold"] >= scores[89]
    assert certificate["accepted"] == round(certificate["coverage"] * scores.size)


def test_certificate_refuses_when_data_is_thin():
    scores = np.linspace(0.0, 1.0, 10)
    correct = np.asarray([1.0] * 5 + [0.0] * 5)
    certificate = selective_certificate(
        scores, correct, alpha=0.01, delta=0.05, min_coverage=0.0
    )

    assert certificate["coverage"] == 0.0
    assert certificate["threshold"] == -1.0
    assert certificate["empirical_risk"] is None
    assert certificate["upper_bound"] == 1.0


def test_certificate_respects_min_coverage():
    scores, correct = _dataset()
    certificate = selective_certificate(
        scores, correct, alpha=0.1, delta=0.05, min_coverage=0.99
    )
    # 100% покрытие включает все ошибки и не проходит по границе
    assert certificate["coverage"] == 0.0


def test_certificate_rejects_invalid_input():
    scores, correct = _dataset()
    with pytest.raises(ValueError):
        selective_certificate(scores, correct, alpha=1.5)
    with pytest.raises(ValueError):
        selective_certificate(scores, correct, delta=0.0)
    with pytest.raises(ValueError):
        selective_certificate(scores, correct, min_coverage=-0.1)
    with pytest.raises(ValueError):
        selective_certificate(scores + 1.0, correct)
    with pytest.raises(ValueError):
        selective_certificate(scores, correct[:5])
    with pytest.raises(ValueError):
        selective_certificate([], [])


def test_certificate_roundtrip_and_validation(tmp_path):
    scores, correct = _dataset()
    certificate = selective_certificate(scores, correct, alpha=0.1, delta=0.05)
    path = save_certificate(certificate, tmp_path / "conformal.json")
    loaded = load_certificate(path)

    assert loaded == certificate
    assert json.loads(path.read_text(encoding="utf-8"))["schema_version"] == "1"

    broken = dict(certificate)
    broken["schema_version"] = "2"
    with pytest.raises(ValueError):
        validate_certificate(broken)

    broken = dict(certificate)
    broken["method"] = "magic"
    with pytest.raises(ValueError):
        validate_certificate(broken)

    broken = dict(certificate)
    broken["threshold"] = 2.0
    with pytest.raises(ValueError):
        validate_certificate(broken)

    broken = dict(certificate)
    broken["coverage"] = 1.5
    with pytest.raises(ValueError):
        validate_certificate(broken)

    broken = dict(certificate)
    broken["gate"] = "ok"
    with pytest.raises(TypeError):
        validate_certificate(broken)


def test_certificate_gives_per_question_guarantee():
    rng = np.random.default_rng(0)
    n = 400
    scores = rng.uniform(0.0, 1.0, n)
    correct = (rng.uniform(0.0, 1.0, n) > scores).astype(float)
    groups = np.asarray(["a", "b"] * (n // 2))

    joint = selective_certificate(
        scores, correct, groups=groups, alpha=0.2, delta=0.1, min_coverage=0.0
    )
    pooled = selective_certificate(
        scores, correct, alpha=0.2, delta=0.1, min_coverage=0.0
    )

    assert set(joint["groups"]) == {"a", "b"}
    assert joint["groups"]["a"]["delta"] == pytest.approx(0.05)
    for stats in joint["groups"].values():
        assert stats["upper_bound"] <= joint["alpha"]
    assert joint["upper_bound"] == pytest.approx(
        max(stats["upper_bound"] for stats in joint["groups"].values())
    )
    # гарантия по каждому вопросу не может быть шире общей
    assert joint["coverage"] <= pooled["coverage"]


def test_certificate_rejects_bad_groups():
    scores, correct = _dataset()
    with pytest.raises(ValueError):
        selective_certificate(scores, correct, groups=["a", "b"])

    certificate = selective_certificate(scores, correct)
    broken = dict(certificate)
    broken["groups"] = ["a", "b"]
    with pytest.raises(TypeError):
        validate_certificate(broken)
