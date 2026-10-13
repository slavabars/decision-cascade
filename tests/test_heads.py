"""Утверждения для голов решений — написаны до реализации (TDD RED)."""
import pytest
import torch

from src.decision_head.heads import ChoiceHead, DecisionHeads, NoulHead, ScoreHead
from src.sampler.questions import ChoiceSpec, NoulSpec, ScoreSpec


def _features(dim: int, *values: float) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float32).reshape(1, -1) if values else torch.zeros(1, dim)


def test_noul_head_high_feature_towards_one():
    head = NoulHead(dim=2)
    head.linear.weight.data = torch.tensor([[4.0, -4.0]])
    head.linear.bias.data = torch.tensor([0.0])
    res = head(torch.tensor([[1.0, -1.0]]))
    assert res.type == "noul"
    assert 0.0 <= res.noul <= 1.0
    assert res.noul > 0.5
    assert res.confidence == pytest.approx(2.0 * abs(res.noul - 0.5), abs=1e-6)


def test_noul_head_low_feature_towards_zero():
    head = NoulHead(dim=2)
    head.linear.weight.data = torch.tensor([[4.0, -4.0]])
    head.linear.bias.data = torch.tensor([0.0])
    res = head(torch.tensor([[-1.0, 1.0]]))
    assert res.noul < 0.5
    assert res.noul > 0.0


def test_choice_head_distribution_sums_to_one():
    head = ChoiceHead(dim=2, options=("a", "b", "c"))
    head.linear.weight.data = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
    head.linear.bias.data = torch.zeros(3)
    res = head(torch.tensor([[2.0, 1.0]]))
    assert res.type == "choice"
    assert set(res.probabilities) == {"a", "b", "c"}
    assert abs(sum(res.probabilities.values()) - 1.0) < 1e-6
    assert res.choice in {"a", "b", "c"}
    assert res.confidence == pytest.approx(max(res.probabilities.values()), abs=1e-6)


def test_choice_head_argmax_matches_selected():
    head = ChoiceHead(dim=1, options=("first", "second"))
    head.linear.weight.data = torch.tensor([[5.0], [-5.0]])
    head.linear.bias.data = torch.zeros(2)
    res = head(torch.tensor([[1.0]]))
    assert res.choice == "first"
    assert res.probabilities["first"] > res.probabilities["second"]


def test_score_head_weighted_mean():
    head = ScoreHead(dim=1, levels=("low", "mid", "high"))
    head.linear.weight.data = torch.tensor([[2.0], [3.0], [4.0]])
    head.linear.bias.data = torch.zeros(3)
    res = head(torch.tensor([[1.0]]))
    assert res.type == "score"
    assert res.legend == ("low", "mid", "high")
    assert abs(sum(res.probabilities.values()) - 1.0) < 1e-6
    expected = (
        0.0 * res.probabilities["low"]
        + 1.0 * res.probabilities["mid"]
        + 2.0 * res.probabilities["high"]
    )
    assert res.score == pytest.approx(expected, abs=1e-6)
    assert 0.0 <= res.score <= 2.0


def test_score_head_uniform_distribution_mid_score():
    head = ScoreHead(dim=1, levels=("low", "high"))
    head.linear.weight.data = torch.zeros(2, 1)
    head.linear.bias.data = torch.zeros(2)
    res = head(torch.tensor([[0.0]]))
    assert res.score == pytest.approx(0.5, abs=1e-6)
    assert res.confidence == pytest.approx(0.5, abs=1e-6)


def test_decision_heads_picks_right_head_per_question():
    dm = DecisionHeads(
        dim=1,
        questions={
            "u": NoulSpec(instructions="urgent?"),
            "c": ChoiceSpec(instructions="who?", options=("x", "y")),
            "s": ScoreSpec(instructions="level?", levels=("low", "high")),
        },
    )
    assert isinstance(dm.heads["u"], NoulHead)
    assert isinstance(dm.heads["c"], ChoiceHead)
    assert isinstance(dm.heads["s"], ScoreHead)


def test_decision_heads_rejects_duplicate_options():
    with pytest.raises(ValueError, match="unique"):
        DecisionHeads(dim=1, questions={"c": ChoiceSpec("?", ("a", "a"))})


def test_decision_heads_rejects_duplicate_levels():
    with pytest.raises(ValueError, match="unique"):
        DecisionHeads(dim=1, questions={"s": ScoreSpec("?", ("a", "a", "b"))})


def test_decision_heads_batch_single_pass():
    dm = DecisionHeads(
        dim=2,
        questions={
            "n1": NoulSpec(instructions="?",),
            "n2": NoulSpec(instructions="?",),
        },
    )
    for i, head in enumerate(dm.heads.values()):
        head.linear.weight.data[0, 0] = 1.0
        head.linear.weight.data[0, 1] = float(i)
        head.linear.bias.data.fill_(0.0)
    features = torch.tensor([[1.0, 0.0], [0.0, 2.0]], dtype=torch.float32)
    out = dm.evaluate(features)
    assert set(out) == {"n1", "n2"}
    for res in out.values():
        assert res.type == "noul"
    assert out["n1"].noul != out["n2"].noul