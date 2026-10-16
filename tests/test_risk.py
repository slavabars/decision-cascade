"""Тесты оценщика риска: признаки решения, P(ошибка), сохранение/загрузка, обучение."""
import numpy as np
import pytest

from src.risk import (
    FEATURE_NAMES,
    LEGACY_FEATURE_NAMES,
    ErrorRiskModel,
    decision_distribution,
    decision_features,
    features_matrix,
)
from src.schemas import ChoiceResult, NoulResult, ScoreResult


def _noul(p: float) -> NoulResult:
    return NoulResult(noul=p, confidence=abs(2 * p - 1))


def _choice(probs: dict[str, float]) -> ChoiceResult:
    best = max(probs, key=probs.__getitem__)
    return ChoiceResult(
        choice=best, probabilities=probs, confidence=probs[best]
    )


def test_distribution_noul_is_binary():
    dist = decision_distribution(_noul(0.8))
    assert dist.tolist() == pytest.approx([0.8, 0.2])


def test_features_confident_choice_has_low_entropy():
    features = decision_features(_choice({"a": 0.97, "b": 0.02, "c": 0.01}))
    assert features.shape == (len(FEATURE_NAMES),)
    confidence, entropy, margin = features[:3]
    assert confidence == pytest.approx(0.97)
    assert entropy < 0.2
    assert margin == pytest.approx(0.95)


def test_features_uniform_choice_has_max_entropy():
    features = decision_features(_choice({"a": 0.5, "b": 0.5}))
    assert features[0] == pytest.approx(0.5)
    assert features[1] == pytest.approx(1.0)  # нормированная энтропия
    assert features[2] == pytest.approx(0.0)


def test_features_score_uses_probabilities():
    result = ScoreResult(
        score=1.0,
        probabilities={"low": 0.1, "mid": 0.1, "high": 0.8},
        confidence=0.8,
        legend=["low", "mid", "high"],
    )
    features = decision_features(result)
    assert features[0] == pytest.approx(0.8)


def test_features_log_odds_gap_is_raw_logit_gap():
    # ln(p1/p2) для k >= 3 — это разрыв сырых логитов до softmax
    features = decision_features(_choice({"a": 0.97, "b": 0.02, "c": 0.01}))
    assert features[3] == pytest.approx(np.log(0.97 / 0.02))
    assert features[4] == pytest.approx(3.0)  # n_classes


def test_features_noul_logit_keeps_sign():
    left = decision_features(_noul(0.1))
    right = decision_features(_noul(0.9))
    # первые три признака симметричны относительно p <-> 1-p, знак — нет
    assert right[5] == pytest.approx(np.log(0.9 / 0.1))
    assert left[5] == pytest.approx(-right[5])
    assert left[4] == pytest.approx(2.0)  # бинарный исход
    assert left[0] == pytest.approx(right[0])  # уверенность одинакова


def test_legacy_model_uses_only_named_features():
    legacy = ErrorRiskModel(
        weights=np.array([-6.0, 0.0, 0.0]),
        bias=3.0,
        feature_names=LEGACY_FEATURE_NAMES,
    )
    full = ErrorRiskModel(
        weights=np.array([-6.0, 0.0, 0.0, 0.0, 0.0, 0.0]), bias=3.0
    )
    for decision in (_noul(0.9), _noul(0.1), _choice({"a": 0.5, "b": 0.5})):
        assert legacy.predict_error(decision) == pytest.approx(
            full.predict_error(decision)
        )
    # знаковый логит различает 0.9 и 0.1, но на модель старого формата не влияет
    assert legacy.predict_error(_noul(0.9)) == pytest.approx(
        legacy.predict_error(_noul(0.1))
    )
    signed = ErrorRiskModel(
        weights=np.array([-6.0, 0.0, 0.0, 0.0, 0.0, -2.0]), bias=3.0
    )
    assert signed.predict_error(_noul(0.9)) != pytest.approx(
        signed.predict_error(_noul(0.1))
    )


def test_unknown_feature_name_rejected():
    with pytest.raises(ValueError, match="unknown feature_names"):
        ErrorRiskModel(weights=np.array([1.0]), bias=0.0, feature_names=("x",))


def test_features_matrix_stacks_rows():
    matrix = features_matrix([_noul(0.9), _noul(0.1)])
    assert matrix.shape == (2, len(FEATURE_NAMES))


def test_predict_error_monotone_in_confidence():
    # вес на уверенность отрицательный: выше уверенность -> ниже P(ошибка)
    model = ErrorRiskModel(
        weights=np.array([-6.0, 0.0, 0.0]),
        bias=3.0,
        feature_names=LEGACY_FEATURE_NAMES,
    )
    uncertain = _noul(0.55)  # близко к 0.5 -> уверенность около 0.55
    confident = _noul(0.95)
    low_error = model.predict_error(confident)
    high_error = model.predict_error(uncertain)
    assert 0.0 <= low_error < high_error <= 1.0
    assert model.should_escalate(uncertain, risk_budget=0.2)
    assert not model.should_escalate(confident, risk_budget=0.2)


def test_save_load_roundtrip(tmp_path):
    # модель старого формата (три признака) переживает сохранение и загрузку
    model = ErrorRiskModel(
        weights=np.array([-6.0, 1.5, 0.5]),
        bias=3.0,
        feature_names=LEGACY_FEATURE_NAMES,
    )
    path = model.save(tmp_path / "nested" / "risk.json")
    assert path.exists()
    restored = ErrorRiskModel.load(path)
    assert restored.weights.tolist() == pytest.approx(model.weights.tolist())
    assert restored.bias == pytest.approx(model.bias)
    assert restored.feature_names == model.feature_names


def test_weights_size_mismatch_rejected():
    with pytest.raises(ValueError):
        ErrorRiskModel(weights=np.array([1.0, 2.0]), bias=0.0)


def test_fit_error_risk_model_learns_from_decisions():
    from training.recipe.risk import fit_error_risk_model

    decisions = [_noul(p) for p in (0.95, 0.9, 0.85, 0.1, 0.15, 0.2)]
    correct = [True, True, True, False, False, False]
    model = fit_error_risk_model(decisions, correct, seed=0)
    assert 0.0 <= model.predict_error(_noul(0.95)) <= 1.0
    # уверенные решения должны эскалироваться реже неуверенных
    assert model.predict_error(_noul(0.95)) < model.predict_error(_noul(0.1))


def test_fit_single_class_returns_baseline():
    from training.recipe.risk import fit_error_risk_model

    decisions = [_noul(p) for p in (0.9, 0.8, 0.7)]
    model = fit_error_risk_model(decisions, [True, True, True], seed=0)
    assert model.predict_error(_noul(0.9)) < 0.01
