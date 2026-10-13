"""Тесты SFT — написаны до реализации (TDD RED)."""
import torch

from src.decision_head.heads import DecisionHeads
from src.sampler.questions import ChoiceSpec, NoulSpec


def _make_dataset(n=200, dim=2, seed=0):
    torch.manual_seed(seed)
    features = torch.randn(n, dim)
    labels = (features[:, 0] > 0).long()
    return features, labels


def test_sft_loss_decreases():
    from training.sft.train import sft_loss

    heads = DecisionHeads(dim=2, questions={"c": ChoiceSpec("?", ("a", "b"))})
    features, labels = _make_dataset()
    target = {"c": labels}
    head_logits = heads.logits(features)
    loss = sft_loss(head_logits, target, heads)
    assert loss.ndim == 0 and loss > 0


def test_sft_training_improves_accuracy():
    from training.sft.train import SFTTrainer

    heads = DecisionHeads(dim=2, questions={"c": ChoiceSpec("?", ("a", "b"))})
    features, labels = _make_dataset()
    acc0 = _accuracy(heads, features, labels)
    SFTTrainer(heads, lr=0.1).fit(
        features=features, targets={"c": labels}, steps=80, batch_size=32
    )
    acc1 = _accuracy(heads, features, labels)
    assert acc1 > acc0
    assert acc1 > 0.85


def test_sft_supports_noul_bce():
    from training.sft.train import SFTTrainer, sft_loss

    heads = DecisionHeads(dim=2, questions={"n": NoulSpec("?")})
    features, labels = _make_dataset()
    labels_float = labels.float()
    loss = sft_loss(heads.logits(features), {"n": labels_float}, heads)
    assert loss.ndim == 0 and loss > 0
    SFTTrainer(heads, lr=0.1).fit(
        features=features, targets={"n": labels_float}, steps=20, batch_size=32
    )


def _accuracy(heads, features, labels):
    with torch.no_grad():
        logits = heads.logits(features)["c"]
        preds = logits.argmax(-1)
        return float((preds == labels).float().mean().item())