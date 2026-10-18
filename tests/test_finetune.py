"""Тесты дообучения энкодера: путь с градиентом, цели и тренер.

Модели не загружаются: и энкодеры, и стенд тренера работают на крошечных
подставных модулях, поэтому тесты идут в обычном наборе (не ``-m model``).
"""
from pathlib import Path

import pytest
import torch
from torch import nn

from src.decision_head.heads import DecisionHeads
from src.encoder.base import TrainableStateEncoder
from src.sampler.questions import ChoiceSpec, NoulSpec, ScoreSpec
from training.finetune.train import (
    FinetuneTrainer,
    build_prompts,
    error_targets,
    targets_for,
)

# --- подставные модели -------------------------------------------------------


class _FakeOutput:
    """Вывод «модели» в двух формах сразу: causal и encoder-only."""

    def __init__(self, hidden: torch.Tensor):
        self.hidden_states = [None, hidden]
        self.last_hidden_state = hidden


class _TinyModel(nn.Module):
    """Крошечная «модель»: скрытое состояние зависит от обучаемых весов."""

    def __init__(self, dim: int = 4, vocab: int = 8):
        super().__init__()
        self.embedding = nn.Embedding(vocab, dim)
        self.shift = nn.Parameter(torch.zeros(dim))

    def forward(self, input_ids=None, attention_mask=None, output_hidden_states=False):
        return _FakeOutput(self.embedding(input_ids) + self.shift)

    def save_pretrained(self, path):
        torch.save(self.state_dict(), Path(path) / "model.pt")
        return str(path)


class _FakeTokenizer:
    def save_pretrained(self, path):
        (Path(path) / "tokenizer.txt").write_text("ok", encoding="utf-8")
        return str(path)

    def __call__(self, states, **kwargs):
        width = max(1, min(4, max((len(state) for state in states), default=1)))
        ids = torch.zeros(len(states), width, dtype=torch.long)
        for i, text in enumerate(states):
            for j, char in enumerate(text[:width]):
                ids[i, j] = (ord(char) % 7) + 1
        return {"input_ids": ids, "attention_mask": torch.ones(len(states), width)}


def _encoder_class(kind: str):
    if kind == "causal":
        from src.encoder.gemma import GemmaStateEncoder

        return GemmaStateEncoder
    from src.encoder.transformer import TransformerStateEncoder

    return TransformerStateEncoder


def _make_encoder(kind: str, dim: int = 4):
    encoder = _encoder_class(kind)(feature_dim=dim, device="cpu", max_length=None)
    encoder._model = _TinyModel(dim)
    encoder._tokenizer = _FakeTokenizer()
    return encoder


# --- контракт дообучаемого энкодера -----------------------------------------


@pytest.mark.parametrize("kind", ["causal", "encoder"])
def test_encoder_is_trainable_state_encoder(kind):
    assert isinstance(_make_encoder(kind), TrainableStateEncoder)


@pytest.mark.parametrize("kind", ["causal", "encoder"])
def test_encode_trainable_keeps_graph(kind):
    encoder = _make_encoder(kind)
    encoder.set_trainable(True)
    features = encoder.encode_trainable(["ab", "cd"])
    assert features.requires_grad
    features.sum().backward()
    assert encoder._model.embedding.weight.grad is not None
    assert encoder._model.shift.grad is not None


@pytest.mark.parametrize("kind", ["causal", "encoder"])
def test_encode_stays_inference_path(kind):
    """Инференс-путь не должен ни тянуть граф, ни держать признаки на GPU."""
    encoder = _make_encoder(kind)
    encoder.set_trainable(True)
    features = encoder.encode(["ab", "cd"])
    assert not features.requires_grad
    assert features.device.type == "cpu"


@pytest.mark.parametrize("kind", ["causal", "encoder"])
def test_encode_trainable_matches_encode_in_eval_mode(kind):
    """Оба пути обязаны совпадать: дообучение идёт на тех же признаках, что и замер."""
    encoder = _make_encoder(kind)
    encoder.set_trainable(False)
    plain = encoder.encode(["ab", "cd", "abc"])
    trainable = encoder.encode_trainable(["ab", "cd", "abc"])
    assert torch.allclose(plain, trainable.detach().cpu())


@pytest.mark.parametrize("kind", ["causal", "encoder"])
def test_set_trainable_toggles_parameters_and_mode(kind):
    encoder = _make_encoder(kind)
    encoder.set_trainable(True)
    assert encoder._model.training is True
    assert len(encoder.trainable_parameters()) == 2
    encoder.set_trainable(False)
    assert encoder._model.training is False
    assert encoder.trainable_parameters() == []


@pytest.mark.parametrize("kind", ["causal", "encoder"])
def test_encoder_methods_require_load(kind):
    encoder = _encoder_class(kind)(feature_dim=4, device="cpu")
    with pytest.raises(RuntimeError):
        encoder.encode_trainable(["a"])
    with pytest.raises(RuntimeError):
        encoder.set_trainable(True)
    with pytest.raises(RuntimeError):
        encoder.trainable_parameters()
    with pytest.raises(RuntimeError):
        encoder.save_pretrained("nowhere")


@pytest.mark.parametrize("kind", ["causal", "encoder"])
def test_save_pretrained_writes_model_and_tokenizer(kind, tmp_path):
    encoder = _make_encoder(kind)
    target = tmp_path / "encoder"
    assert encoder.save_pretrained(target) == target
    assert (target / "model.pt").exists()
    assert (target / "tokenizer.txt").exists()


# --- цели --------------------------------------------------------------------


def test_targets_match_benchmark_builder():
    """Цели дообучения и бенчмарка обязаны совпадать, иначе метрики не сравнимы."""
    from eval.benchmark.core import _build_targets

    noul = NoulSpec(instructions="Is it?")
    choice = ChoiceSpec(instructions="Which?", options=("a", "b", "c"))
    score = ScoreSpec(instructions="How big?", levels=("low", "mid", "high"))
    for spec, labels in (
        (noul, [1.0, 0.0, 1.0]),
        (choice, ["c", "a", "b"]),
        (score, [2, 0, 1]),
    ):
        assert torch.equal(
            targets_for(spec, labels), _build_targets(spec, labels)
        )


def test_targets_for_rejects_unknown_spec():
    class _UnknownSpec:
        pass

    with pytest.raises(TypeError):
        targets_for(_UnknownSpec(), [1])


def test_error_targets_noul_uses_taken_action():
    logits = torch.tensor([2.0, -2.0, -0.5])
    labels = torch.tensor([1.0, 1.0, 0.0])
    assert torch.equal(error_targets(logits, labels), torch.tensor([0.0, 1.0, 0.0]))


def test_error_targets_choice_uses_argmax():
    logits = torch.tensor([[3.0, 0.0], [0.0, 1.0]])
    labels = torch.tensor([0, 0])
    assert torch.equal(error_targets(logits, labels), torch.tensor([0.0, 1.0]))


def test_build_prompts_one_per_question_and_example():
    specs = {
        "flag": NoulSpec(instructions="Is the flag raised?"),
        "size": ScoreSpec(instructions="How big?", levels=("s", "m")),
    }
    prompts = build_prompts(specs, _examples(3))
    assert set(prompts) == {"flag", "size"}
    assert all(len(group) == 3 for group in prompts.values())
    assert "State: yes alpha" in prompts["flag"][0]
    assert "Scale: s, m" in prompts["size"][0]


# --- тренер ------------------------------------------------------------------


class _StubEncoder:
    """Дообучаемый стенд: признаки — обучаемая функция от текста состояния."""

    def __init__(self, dim: int = 4):
        self.model = nn.Linear(dim, dim)
        self.trainable: bool | None = None

    @staticmethod
    def _base(states: list[str], dim: int) -> torch.Tensor:
        """Признаки состояния: O(1) по масштабу, иначе ``tanh`` насыщается."""
        base = torch.zeros(len(states), dim)
        for i, text in enumerate(states):
            for j, char in enumerate(text):
                base[i, j % dim] += float(ord(char) % 5)
            base[i] /= max(1, len(text))
        return base

    def encode_trainable(self, states: list[str]) -> torch.Tensor:
        return torch.tanh(self.model(self._base(states, self.model.in_features)))

    def encode(self, states: list[str]) -> torch.Tensor:
        return self.encode_trainable(states).detach().cpu()

    def set_trainable(self, flag: bool = True) -> None:
        self.trainable = flag
        for param in self.model.parameters():
            param.requires_grad_(flag)

    def trainable_parameters(self) -> list[torch.Tensor]:
        return [p for p in self.model.parameters() if p.requires_grad]


def _specs() -> dict:
    return {"flag": NoulSpec(instructions="Is the flag raised?")}


def _examples(n: int = 32) -> list[dict]:
    """Строки «yes/no» выучиваются, «mix» — нет: у одной и той же строки обе метки.

    Противоречивые строки нужны фазе ошибки: без них головы обучаются идеально,
    ошибок нет, и предсказывать нечего.
    """
    states = ("yes alpha", "no beta", "mix gamma", "mix delta")
    examples = []
    for i in range(n):
        state = states[i % len(states)]
        if state.startswith("mix"):
            label = float((i // len(states)) % 2 == 0)
        else:
            label = float(state.startswith("yes"))
        examples.append({"state": state, "labels": {"flag": label}})
    return examples


def test_finetune_fit_moves_encoder_and_heads():
    encoder = _StubEncoder(dim=4)
    heads = DecisionHeads(4, _specs(), seed=0)
    before_encoder = encoder.model.weight.detach().clone()
    before_head = heads.heads["flag"].linear.weight.detach().clone()

    trainer = FinetuneTrainer(encoder, heads, lr=0.05, seed=0)
    losses = trainer.fit(_specs(), _examples(), steps=60, batch_size=8)

    assert sum(losses[-10:]) < sum(losses[:10])
    assert not torch.equal(encoder.model.weight, before_encoder)
    assert not torch.equal(heads.heads["flag"].linear.weight, before_head)
    assert encoder.trainable is True
    # головы обязаны оказаться на устройстве признаков энкодера
    assert next(heads.parameters()).device == next(encoder.model.parameters()).device


def test_finetune_fit_error_trains_encoder_and_freezes_heads():
    encoder = _StubEncoder(dim=4)
    heads = DecisionHeads(4, _specs(), seed=0)
    trainer = FinetuneTrainer(encoder, heads, lr=0.05, seed=0)
    trainer.fit(_specs(), _examples(), steps=40, batch_size=8)

    before_encoder = encoder.model.weight.detach().clone()
    before_head = heads.heads["flag"].linear.weight.detach().clone()
    trainer.fit_error(_specs(), _examples(), steps=30, batch_size=8, lr=0.5)

    assert trainer.error_head is not None
    assert (trainer.error_head.in_features, trainer.error_head.out_features) == (4, 1)
    # градиент фазы ошибки доходит именно до энкодера...
    assert encoder.model.weight.grad is not None
    assert float(encoder.model.weight.grad.abs().sum()) > 0.0
    assert not torch.equal(encoder.model.weight, before_encoder)
    # ...а головы остаются ровно такими, какими их заморозили
    assert torch.equal(heads.heads["flag"].linear.weight, before_head)
    assert all(not param.requires_grad for param in heads.parameters())
    assert heads.training is False
