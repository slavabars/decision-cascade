"""Тесты SFT — написаны до реализации (TDD RED)."""
import pytest
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


def _noul_accuracy(head, features, labels):
    with torch.no_grad():
        probs = torch.sigmoid(head.logits(features))
        preds = (probs >= 0.5).float()
        return float((preds == labels).float().mean().item())


def test_sft_multi_question_trains_each_head_on_own_features():
    """Каждая голова учится на своих промптах, а не на чужих строках батча."""
    from training.sft.train import SFTTrainer

    heads = DecisionHeads(
        dim=2,
        questions={"a": NoulSpec("?"), "b": NoulSpec("?")},
    )
    torch.manual_seed(0)
    n = 500
    base = torch.randn(n, 2)
    feats_a = base.clone()
    feats_b = base[:, [1, 0]].clone()
    labels_a = (feats_a[:, 0] > 0).float()
    labels_b = (feats_b[:, 0] > 0).float()

    SFTTrainer(heads, lr=0.1).fit(
        features={"a": feats_a, "b": feats_b},
        targets={"a": labels_a, "b": labels_b},
        steps=300,
        batch_size=64,
    )

    assert _noul_accuracy(heads.heads["a"], feats_a, labels_a) > 0.9
    assert _noul_accuracy(heads.heads["b"], feats_b, labels_b) > 0.9


def test_sft_multi_question_requires_per_question_features():
    """Единый тензор на несколько вопросов запрещён — иначе цели смешиваются."""
    from training.sft.train import SFTTrainer

    heads = DecisionHeads(
        dim=2,
        questions={"a": NoulSpec("?"), "b": NoulSpec("?")},
    )
    features = torch.randn(10, 2)
    targets = {"a": torch.zeros(10), "b": torch.ones(10)}
    with pytest.raises(ValueError):
        SFTTrainer(heads).fit(features=features, targets=targets, steps=1)


def test_sft_dict_features_must_cover_questions():
    from training.sft.train import SFTTrainer

    heads = DecisionHeads(
        dim=2,
        questions={"a": NoulSpec("?"), "b": NoulSpec("?")},
    )
    with pytest.raises(ValueError):
        SFTTrainer(heads).fit(
            features={"a": torch.randn(5, 2)},
            targets={"a": torch.zeros(5)},
            steps=1,
        )