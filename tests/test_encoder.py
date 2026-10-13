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


class _RecordingTokenizer:
    def __init__(self):
        self.kwargs: dict = {}

    def __call__(self, states, **kwargs):
        self.kwargs = kwargs
        n = len(states)
        return {
            "input_ids": torch.zeros(n, 4, dtype=torch.long),
            "attention_mask": torch.ones(n, 4),
        }


def test_gemma_encode_truncates_to_max_length():
    from src.encoder.gemma import GemmaStateEncoder

    enc = GemmaStateEncoder(feature_dim=4, max_length=128)
    tok = _RecordingTokenizer()
    enc._model = _FakeModel(hidden_last=torch.randn(1, 4, 4))
    enc._tokenizer = tok
    enc.encode(["a long state"])
    assert tok.kwargs["truncation"] is True
    assert tok.kwargs["max_length"] == 128


def test_gemma_encode_can_disable_truncation():
    from src.encoder.gemma import GemmaStateEncoder

    enc = GemmaStateEncoder(feature_dim=4, max_length=None)
    tok = _RecordingTokenizer()
    enc._model = _FakeModel(hidden_last=torch.randn(1, 4, 4))
    enc._tokenizer = tok
    enc.encode(["a state"])
    assert tok.kwargs["truncation"] is False
    assert tok.kwargs.get("max_length") is None


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


class _BatchRecordingModel:
    def __init__(self, dim=4):
        self.dim = dim
        self.sizes: list[int] = []

    def __call__(self, **kwargs):
        n = kwargs["input_ids"].shape[0]
        self.sizes.append(n)
        return _FakeOutput(torch.zeros(n, 1, self.dim))

    def eval(self):
        pass


def test_gemma_encode_chunks_by_batch_size():
    from src.encoder.gemma import GemmaStateEncoder

    enc = GemmaStateEncoder(feature_dim=4, batch_size=2)
    model = _BatchRecordingModel(4)
    enc._model, enc._tokenizer = model, _FakeTokenizer()
    out = enc.encode(["a", "b", "c", "d", "e"])
    assert model.sizes == [2, 2, 1]
    assert tuple(out.shape) == (5, 4)


def test_gemma_encode_empty_returns_zero_rows():
    from src.encoder.gemma import GemmaStateEncoder

    enc = GemmaStateEncoder(feature_dim=4)
    enc._model, enc._tokenizer = _FakeModel(torch.randn(0, 1, 4)), _FakeTokenizer()
    out = enc.encode([])
    assert tuple(out.shape) == (0, 4)


def test_gemma_encoder_rejects_bad_batch_size():
    from src.encoder.gemma import GemmaStateEncoder

    with pytest.raises(ValueError):
        GemmaStateEncoder(batch_size=0)