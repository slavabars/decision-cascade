"""Тесты публичного адаптера (AG News) и помощников отчёта — без сети."""
import numpy as np
import torch

from eval.datasets.public_dataset import (
    AG_NEWS_LABELS,
    PUBLIC_QUESTIONS,
    PUBLIC_SPECS,
    _tech_level,
    map_example,
)
from src.sampler.questions import ChoiceSpec, NoulSpec, ScoreSpec


def test_public_specs_types_and_options():
    assert isinstance(PUBLIC_SPECS["topic"], ChoiceSpec)
    assert isinstance(PUBLIC_SPECS["business"], NoulSpec)
    assert isinstance(PUBLIC_SPECS["tech_level"], ScoreSpec)
    assert list(PUBLIC_SPECS["topic"].options) == list(AG_NEWS_LABELS)
    assert PUBLIC_QUESTIONS["topic"]["options"] == list(AG_NEWS_LABELS)


def test_map_example_business_label():
    ex = map_example("train-000000", "Stocks rally", 2)
    assert ex["labels"]["topic"] == "Business"
    assert ex["labels"]["business"] == 1
    assert ex["labels"]["tech_level"] == 1


def test_map_example_sci_tech_label():
    ex = map_example("train-000001", "New GPU", 3)
    assert ex["labels"]["topic"] == "Sci/Tech"
    assert ex["labels"]["business"] == 0
    assert ex["labels"]["tech_level"] == 2


def test_map_example_world_label_is_low_tech():
    ex = map_example("train-000002", "Election results", 0)
    assert ex["labels"]["business"] == 0
    assert ex["labels"]["tech_level"] == 0


def test_map_example_rejects_unknown_label():
    import pytest

    with pytest.raises(ValueError):
        map_example("x", "text", 99)


def test_tech_level_mapping_is_ordinal():
    assert [_tech_level(i) for i in range(4)] == [0, 0, 1, 2]


class _FakeNoulHead:
    def logits(self, features):
        return features[:, 0]


class _FakeChoiceHead:
    def logits(self, features):
        return features[:, :3]


def test_report_labels_for_each_type():
    from eval.report.run import _labels

    examples = [
        {"labels": {"u": 1, "topic": "Business", "s": 2}},
        {"labels": {"u": 0, "topic": "World", "s": 0}},
    ]
    assert list(_labels(NoulSpec("u?"), examples, "u")) == [1.0, 0.0]
    assert list(_labels(PUBLIC_SPECS["topic"], examples, "topic")) == [2, 0]
    assert list(_labels(PUBLIC_SPECS["tech_level"], examples, "s")) == [2, 0]


def test_report_confidence_and_correct_shapes():
    from eval.report.run import _confidence_and_correct

    features = torch.randn(4, 3)
    confidence, correct = _confidence_and_correct(
        PUBLIC_SPECS["topic"], _FakeChoiceHead(), features, np.array([0, 1, 2, 0])
    )
    assert confidence.shape == (4,)
    assert ((confidence >= 0) & (confidence <= 1)).all()
    assert set(np.unique(correct)).issubset({0.0, 1.0})

    noul_conf, noul_correct = _confidence_and_correct(
        PUBLIC_SPECS["business"], _FakeNoulHead(), torch.randn(4, 2), np.array([1, 0, 1, 0])
    )
    assert noul_conf.shape == (4,)
    assert set(np.unique(noul_correct)).issubset({0.0, 1.0})
