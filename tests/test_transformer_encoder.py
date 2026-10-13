"""Тесты encoder-only трансформера, общей pooling-логики и фабрики энкодеров."""
import pytest
import torch


def test_transformer_encoder_is_state_encoder():
    from src.encoder.base import StateEncoder
    from src.encoder.transformer import TransformerStateEncoder

    assert isinstance(TransformerStateEncoder(), StateEncoder)


def test_transformer_encoder_requires_load():
    from src.encoder.transformer import TransformerStateEncoder

    enc = TransformerStateEncoder()
    with pytest.raises(RuntimeError):
        enc.encode(["hello"])


def test_transformer_encoder_defaults():
    from src.encoder.transformer import TransformerStateEncoder

    enc = TransformerStateEncoder()
    assert enc.model_id == "bert-base-uncased"
    assert enc.feature_dim == 768


def test_transformer_encoder_rejects_bad_batch_size():
    from src.encoder.transformer import TransformerStateEncoder

    with pytest.raises(ValueError):
        TransformerStateEncoder(batch_size=0)


class _FakeOutput:
    def __init__(self, last_hidden_state=None, hidden_states=None):
        if last_hidden_state is not None:
            self.last_hidden_state = last_hidden_state
        self.hidden_states = hidden_states


class _FakeModel:
    def __init__(self, output):
        self._output = output

    def __call__(self, **kwargs):
        return self._output

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


def test_transformer_encode_truncates_to_max_length():
    from src.encoder.transformer import TransformerStateEncoder

    enc = TransformerStateEncoder(feature_dim=4, max_length=128, device="cpu")
    tok = _RecordingTokenizer()
    enc._model = _FakeModel(_FakeOutput(last_hidden_state=torch.randn(1, 4, 4)))
    enc._tokenizer = tok
    enc.encode(["a long state"])
    assert tok.kwargs["truncation"] is True
    assert tok.kwargs["max_length"] == 128


def test_transformer_encode_can_disable_truncation():
    from src.encoder.transformer import TransformerStateEncoder

    enc = TransformerStateEncoder(feature_dim=4, max_length=None, device="cpu")
    tok = _RecordingTokenizer()
    enc._model = _FakeModel(_FakeOutput(last_hidden_state=torch.randn(1, 4, 4)))
    enc._tokenizer = tok
    enc.encode(["a state"])
    assert tok.kwargs["truncation"] is False
    assert tok.kwargs.get("max_length") is None


def test_transformer_encode_pools_last_hidden_state():
    from src.encoder.transformer import TransformerStateEncoder

    enc = TransformerStateEncoder(feature_dim=4, device="cpu")
    fake = _FakeModel(_FakeOutput(last_hidden_state=torch.randn(2, 4, 4)))
    enc._model, enc._tokenizer = fake, _FakeTokenizer()
    out = enc.encode(["first state", "second state"])
    assert tuple(out.shape) == (2, 4)
    assert out.dtype == torch.float32


def test_transformer_encode_falls_back_to_hidden_states():
    """Некоторые конфигурации не отдают ``last_hidden_state`` — берём [-1]."""
    from src.encoder.transformer import TransformerStateEncoder

    enc = TransformerStateEncoder(feature_dim=4, device="cpu")
    fake = _FakeModel(_FakeOutput(hidden_states=[None, torch.randn(2, 4, 4)]))
    enc._model, enc._tokenizer = fake, _FakeTokenizer()
    out = enc.encode(["a state", "b state"])
    assert tuple(out.shape) == (2, 4)


def test_transformer_encode_projects_dim_mismatch():
    from src.encoder.transformer import TransformerStateEncoder

    enc = TransformerStateEncoder(feature_dim=2, device="cpu")
    fake = _FakeModel(_FakeOutput(last_hidden_state=torch.randn(2, 4, 3)))
    enc._model, enc._tokenizer = fake, _FakeTokenizer()
    out = enc.encode(["one", "two"])
    assert tuple(out.shape) == (2, 2)


class _BatchRecordingModel:
    def __init__(self, dim=4):
        self.dim = dim
        self.sizes: list[int] = []

    def __call__(self, **kwargs):
        n = kwargs["input_ids"].shape[0]
        width = kwargs["input_ids"].shape[1]
        self.sizes.append(n)
        return _FakeOutput(last_hidden_state=torch.zeros(n, width, self.dim))

    def eval(self):
        pass


def test_transformer_encode_chunks_by_batch_size():
    from src.encoder.transformer import TransformerStateEncoder

    enc = TransformerStateEncoder(feature_dim=4, batch_size=2, device="cpu")
    model = _BatchRecordingModel(4)
    enc._model, enc._tokenizer = model, _FakeTokenizer()
    out = enc.encode(["a", "b", "c", "d", "e"])
    assert model.sizes == [2, 2, 1]
    assert tuple(out.shape) == (5, 4)


def test_transformer_encode_empty_returns_zero_rows():
    from src.encoder.transformer import TransformerStateEncoder

    enc = TransformerStateEncoder(feature_dim=4, device="cpu")
    enc._model = _FakeModel(_FakeOutput(last_hidden_state=torch.randn(0, 1, 4)))
    enc._tokenizer = _FakeTokenizer()
    out = enc.encode([])
    assert tuple(out.shape) == (0, 4)


class _FixedMaskTokenizer:
    def __init__(self, mask):
        self._mask = mask

    def __call__(self, states, **kwargs):
        n = len(states)
        return {
            "input_ids": torch.zeros(n, self._mask.shape[1], dtype=torch.long),
            "attention_mask": self._mask.clone(),
        }


def test_transformer_encode_pooling_ignores_padding():
    from src.encoder.transformer import TransformerStateEncoder

    enc = TransformerStateEncoder(feature_dim=2, max_length=None, device="cpu")
    hidden = torch.tensor([[[1.0, 1.0], [3.0, 3.0], [100.0, 100.0], [100.0, 100.0]]])
    enc._model = _FakeModel(_FakeOutput(last_hidden_state=hidden))
    enc._tokenizer = _FixedMaskTokenizer(torch.tensor([[1.0, 1.0, 0.0, 0.0]]))
    out = enc.encode(["state"])
    assert torch.allclose(out, torch.tensor([[2.0, 2.0]]))


class _IdsModel:
    """Признаки = input_ids: pad-позиции (0) тянут обычное среднее вниз."""

    def __call__(self, input_ids=None, **kwargs):
        return _FakeOutput(last_hidden_state=input_ids.float().unsqueeze(-1))

    def eval(self):
        pass


class _LengthTokenizer:
    """Реальный токен = 1, pad = 0; длина = len(state)."""

    def __call__(self, states, **kwargs):
        lengths = [max(1, len(state)) for state in states]
        width = max(lengths)
        ids = torch.zeros(len(states), width, dtype=torch.long)
        mask = torch.zeros(len(states), width)
        for i, length in enumerate(lengths):
            ids[i, :length] = 1
            mask[i, :length] = 1.0
        return {"input_ids": ids, "attention_mask": mask}


def test_transformer_encode_is_batch_invariant():
    from src.encoder.transformer import TransformerStateEncoder

    enc = TransformerStateEncoder(feature_dim=1, max_length=None, device="cpu")
    enc._model, enc._tokenizer = _IdsModel(), _LengthTokenizer()
    alone = enc.encode(["a"])
    batched = enc.encode(["a", "longer state here"])
    assert torch.allclose(alone[0], batched[0])
    assert torch.allclose(alone, torch.ones(1, 1))


def test_create_state_encoder_defaults_to_causal(monkeypatch):
    monkeypatch.delenv("DECISION_ENCODER_KIND", raising=False)
    from src.encoder.factory import create_state_encoder
    from src.encoder.gemma import GemmaStateEncoder

    enc = create_state_encoder("openai-community/gpt2", 768)
    assert isinstance(enc, GemmaStateEncoder)


def test_create_state_encoder_encoder_kind():
    from src.encoder.factory import create_state_encoder
    from src.encoder.transformer import TransformerStateEncoder

    enc = create_state_encoder("bert-base-uncased", 768, kind="encoder")
    assert isinstance(enc, TransformerStateEncoder)


def test_create_state_encoder_reads_env(monkeypatch):
    monkeypatch.setenv("DECISION_ENCODER_KIND", "encoder")
    from src.encoder.factory import create_state_encoder
    from src.encoder.transformer import TransformerStateEncoder

    enc = create_state_encoder("bert-base-uncased", 768)
    assert isinstance(enc, TransformerStateEncoder)


def test_create_state_encoder_rejects_unknown_kind():
    from src.encoder.factory import create_state_encoder

    with pytest.raises(ValueError):
        create_state_encoder("bert-base-uncased", 768, kind="banana")
