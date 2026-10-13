"""Утверждения для схем решений — написаны до реализации (TDD RED)."""
import dataclasses

import pytest

from src.schemas import (
    SCHEMA_VERSION,
    ChoiceResult,
    NoulResult,
    ScoreResult,
    confidence_max,
    confidence_noul,
    normalize_probabilities,
    validate_distribution,
    validate_noul,
)


def test_normalize_sums_to_one():
    dist = normalize_probabilities({"a": 2.0, "b": 1.0})
    assert abs(sum(dist.values()) - 1.0) < 1e-6
    assert dist["a"] > dist["b"] and dist["a"] > 0 and dist["b"] > 0


def test_normalize_returns_copy():
    raw = {"a": 0.6, "b": 0.4}
    out = normalize_probabilities(raw)
    assert out is not raw


def test_normalize_rejects_empty():
    with pytest.raises(ValueError):
        normalize_probabilities({})


def test_normalize_rejects_negative():
    with pytest.raises(ValueError):
        normalize_probabilities({"a": 1.0, "b": -0.1})


def test_normalize_rejects_zero_total():
    with pytest.raises(ValueError):
        normalize_probabilities({"a": 0.0})


def test_validate_noul_bounds():
    assert validate_noul(0.0) == 0.0
    assert validate_noul(0.5) == 0.5
    assert validate_noul(1.0) == 1.0


def test_validate_noul_rejects_out_of_range():
    with pytest.raises(ValueError):
        validate_noul(-0.01)
    with pytest.raises(ValueError):
        validate_noul(1.01)


def test_validate_distribution_accepts_proper_form():
    dist = {"a": 0.4, "b": 0.4, "c": 0.2}
    out = validate_distribution(dist, options=("a", "b", "c"))
    assert abs(sum(out.values()) - 1.0) < 1e-6


def test_validate_distribution_rejects_sum_off():
    with pytest.raises(ValueError):
        validate_distribution({"a": 0.99, "b": 0.0}, options=("a", "b"))


def test_validate_distribution_rejects_missing_option():
    with pytest.raises(ValueError):
        validate_distribution({"a": 1.0}, options=("a", "b"))


def test_validate_distribution_rejects_unknown_option():
    with pytest.raises(ValueError):
        validate_distribution({"a": 1.0, "c": 0.0}, options=("a", "b"))


def test_confidence_noul_edges():
    assert confidence_noul(0.5) == 0.0
    assert confidence_noul(1.0) == 1.0
    assert confidence_noul(0.0) == 1.0
    mid = confidence_noul(0.75)
    assert abs(mid - 0.5) < 1e-6


def test_confidence_max():
    assert confidence_max({"a": 0.8, "b": 0.2}) == 0.8


def test_schema_version_is_string():
    assert isinstance(SCHEMA_VERSION, str) and SCHEMA_VERSION


def test_result_dataclasses_are_frozen_and_typed():
    n = NoulResult(type="noul", noul=0.6, confidence=0.2)
    c = ChoiceResult(
        type="choice",
        choice="billing",
        probabilities={"billing": 0.7, "sales": 0.3},
        confidence=0.7,
    )
    s = ScoreResult(
        type="score",
        score=1.5,
        probabilities={"low": 0.5, "high": 0.5},
        confidence=0.5,
        legend=["low", "high"],
    )
    assert n.type == "noul"
    assert c.type == "choice" and c.choice == "billing"
    assert s.type == "score" and s.legend == ["low", "high"]
    with pytest.raises(dataclasses.FrozenInstanceError):
        n.noul = 0.9