"""Реальные тесты encoder-only трансформера (opt-in, качают модель с HF).

Обычный ``pytest`` их пропускает; прогон на лёгкой BERT:

    DECISION_RUN_MODEL_TESTS=1 .venv/bin/python -m pytest -m model -v

Модель по умолчанию — ``bert-base-uncased`` (~440 МБ), переопределяется через
``DECISION_TEST_EMBED_MODEL``.
"""
import os

import pytest
import torch

from src.encoder.transformer import TransformerStateEncoder

MODEL_ID = os.environ.get("DECISION_TEST_EMBED_MODEL", "bert-base-uncased")
RUN_MODEL_TESTS = os.environ.get("DECISION_RUN_MODEL_TESTS") == "1"

pytestmark = [
    pytest.mark.model,
    pytest.mark.skipif(
        not RUN_MODEL_TESTS,
        reason="set DECISION_RUN_MODEL_TESTS=1 to run real-model tests",
    ),
]


@pytest.fixture(scope="module")
def encoder() -> TransformerStateEncoder:
    return TransformerStateEncoder(
        MODEL_ID, feature_dim=768, max_length=128, device="cpu"
    ).load()


def test_encode_returns_feature_batch(encoder):
    out = encoder.encode(["a short state", "a considerably longer state text here"])
    assert tuple(out.shape) == (2, 768)
    assert out.dtype == torch.float32
    assert torch.isfinite(out).all()


def test_encode_is_batch_invariant(encoder):
    """Один и тот же state в батче 1 и батче 2 даёт одинаковый вектор."""
    alone = encoder.encode(["платёж не проходит, срочно"])
    with_neighbor = encoder.encode(
        ["платёж не проходит, срочно", "совсем другой текст, длиннее в два раза"]
    )
    assert torch.allclose(alone[0], with_neighbor[0], atol=1e-4)


def test_encode_differs_for_different_states(encoder):
    out = encoder.encode(["клиент доволен", "клиент в ярости и угрожает уйти"])
    assert not torch.allclose(out[0], out[1], atol=1e-3)
