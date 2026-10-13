"""Тесты энкодера: интерфейс и контракты без загрузки больших моделей."""
import pytest
import torch


def test_hash_encoder_is_state_encoder():
    from api.server.service import HashStateEncoder
    from src.encoder.base import StateEncoder

    assert isinstance(HashStateEncoder(), StateEncoder)


def test_gemma_encoder_is_state_encoder():
    from src.encoder.base import StateEncoder
    from src.encoder.gemma import GemmaStateEncoder

    assert isinstance(GemmaStateEncoder(), StateEncoder)


def test_gemma_encoder_requires_load():
    from src.encoder.gemma import GemmaStateEncoder

    enc = GemmaStateEncoder()
    with pytest.raises(RuntimeError):
        enc.encode(["hello"])


def test_gemma_encoder_defaults():
    from src.encoder.gemma import GemmaStateEncoder

    enc = GemmaStateEncoder()
    assert enc.model_id == "google/gemma-4-12B"
    assert enc.feature_dim == 768


class _FakeOutput:
    def __init__(self, hidden_last):
        self.hidden_states = [None, hidden_last]


class _FakeModel:
    def __init__(self, hidden_last):
        self._hidden = hidden_last

    def __call__(self, **kwargs):
        return _FakeOutput(self._hidden)

    def eval(self):
        pass


class _FakeTokenizer:
    def __call__(self, states, **kwargs):
        n = len(states)
        return {
            "input_ids": torch.zeros(n, 4, dtype=torch.long),
            "attention_mask": torch.ones(n, 4),
        }


def test_gemma_encode_pools_hidden_states():
    from src.encoder.gemma import GemmaStateEncoder

    enc = GemmaStateEncoder(feature_dim=4)
    fake_model = _FakeModel(hidden_last=torch.randn(2, 4, 4))
    enc._model, enc._tokenizer = fake_model, _FakeTokenizer()
    out = enc.encode(["first state", "second state"])
    assert tuple(out.shape) == (2, 4)
    assert out.dtype == torch.float32


def test_gemma_encode_projects_dim_mismatch():
    from src.encoder.gemma import GemmaStateEncoder

    enc = GemmaStateEncoder(feature_dim=2)
    fake_model = _FakeModel(hidden_last=torch.randn(2, 4, 3))
    enc._model, enc._tokenizer = fake_model, _FakeTokenizer()
    out = enc.encode(["only one state", "and another"])
    assert tuple(out.shape) == (2, 2)