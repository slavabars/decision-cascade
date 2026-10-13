"""Утверждения для параллельного семплера — написаны до реализации (TDD RED)."""
import pytest
import torch

from src.decision_head.heads import DecisionHeads
from src.sampler.parallel import sample_decisions
from src.sampler.questions import ChoiceSpec, NoulSpec, ScoreSpec


def test_sample_all_questions_one_pass():
    dm = DecisionHeads(
        dim=3,
        questions={
            "urgency": NoulSpec(instructions="urgent?"),
            "dept": ChoiceSpec(instructions="who?", options=("billing", "sales")),
            "pain": ScoreSpec(instructions="how bad?", levels=("low", "mid", "high")),
        },
    )
    features = torch.tensor(
        [[1.0, 0.5, -1.0], [0.2, 0.8, 0.4], [-2.0, 0.0, 1.0]], dtype=torch.float32
    )
    results = sample_decisions(dm, features)
    assert set(results) == {"urgency", "dept", "pain"}
    assert results["urgency"].type == "noul"
    assert results["dept"].type == "choice"
    assert results["pain"].type == "score"
    assert 0.0 <= results["urgency"].noul <= 1.0
    assert results["dept"].choice in {"billing", "sales"}
    assert abs(sum(results["dept"].probabilities.values()) - 1.0) < 1e-6
    assert abs(sum(results["pain"].probabilities.values()) - 1.0) < 1e-6


def test_sample_returns_no_raw_logits():
    dm = DecisionHeads(
        dim=1,
        questions={"n": NoulSpec(instructions="?")},
    )
    results = sample_decisions(dm, torch.tensor([[0.0]], dtype=torch.float32))
    payload = results["n"]
    raw = [getattr(payload, a, None) for a in ("logits", "logit", "raw")]
    assert all(v is None for v in raw)


def test_sample_handles_many_parallel_nouls():
    dm = DecisionHeads(
        dim=1,
        questions={f"n{i}": NoulSpec(instructions="?") for i in range(8)},
    )
    for head in dm.heads.values():
        head.linear.weight.data.fill_(1.0)
        head.linear.bias.data.fill_(0.0)
    features = torch.linspace(-3.0, 3.0, 8).reshape(8, 1)
    results = sample_decisions(dm, features)
    assert len(results) == 8
    probs = [results[f"n{i}"].noul for i in range(8)]
    assert probs == sorted(probs)


def test_sample_batch_shape_mismatch_raises():
    dm = DecisionHeads(
        dim=2,
        questions={"n": NoulSpec(instructions="?")},
    )
    with pytest.raises(ValueError):
        sample_decisions(dm, torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float32))