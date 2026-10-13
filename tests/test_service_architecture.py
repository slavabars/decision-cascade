"""Тесты архитектуры сервиса: переиспользование голов, чекпоинты, кэш фич, промпты."""
import torch

from api.server.service import DecisionService
from src.sampler.questions import ChoiceSpec, NoulSpec


class CountingEncoder:
    """Энкодер-шпион: считает вызовы и запоминает переданные состояния."""

    def __init__(self, dim: int = 8):
        self.dim = dim
        self.calls = 0
        self.seen: list[list[str]] = []

    def encode(self, states: list[str]) -> torch.Tensor:
        self.calls += 1
        self.seen.append(list(states))
        return torch.ones(len(states), self.dim)


def test_heads_reused_across_identical_calls():
    svc = DecisionService(seed=0)
    questions = {"u": NoulSpec("urgent?")}
    first = svc.get_heads(64, questions)
    second = svc.get_heads(64, questions)
    assert first is second


def test_distinct_questions_get_distinct_heads():
    svc = DecisionService(seed=0)
    a = svc.get_heads(64, {"u": NoulSpec("urgent?")})
    b = svc.get_heads(64, {"u": NoulSpec("different?")})
    assert a is not b


def test_feature_cache_skips_reencoding_unchanged_state():
    encoder = CountingEncoder(dim=8)
    svc = DecisionService(encoder=encoder, dim=8)
    questions = {
        "u": NoulSpec("urgent?"),
        "c": ChoiceSpec("who?", ("a", "b")),
    }
    svc.decide("state text", questions)
    assert encoder.calls == 1
    svc.decide("state text", questions)
    assert encoder.calls == 1  # всё взято из кэша
    svc.decide("another state", questions)
    assert encoder.calls == 2


def test_instructions_reach_the_encoder():
    encoder = CountingEncoder(dim=8)
    svc = DecisionService(encoder=encoder, dim=8)
    svc.decide("invoice overdue", {"u": NoulSpec("Is it urgent?")})
    prompt = encoder.seen[0][0]
    assert "invoice overdue" in prompt
    assert "Is it urgent?" in prompt


def test_criteria_reach_the_encoder():
    encoder = CountingEncoder(dim=8)
    svc = DecisionService(encoder=encoder, dim=8)
    svc.decide(
        "s",
        {"c": ChoiceSpec("who?", ("billing", "sales"), {"billing": "money", "sales": "deals"})},
    )
    prompt = encoder.seen[0][0]
    assert "money" in prompt and "deals" in prompt


def test_checkpoint_weights_are_loaded_by_service(tmp_path):
    questions = {"u": NoulSpec("urgent?")}

    encoder = CountingEncoder(dim=4)
    svc = DecisionService(encoder=encoder, dim=4, seed=1, checkpoint_dir=tmp_path)
    heads = svc.get_heads(4, questions)
    with torch.no_grad():
        heads.heads["u"].linear.weight.fill_(50.0)
        heads.heads["u"].linear.bias.zero_()
    path = svc.save_heads(4, questions)
    assert path.exists()

    encoder2 = CountingEncoder(dim=4)
    svc2 = DecisionService(encoder=encoder2, dim=4, seed=999, checkpoint_dir=tmp_path)
    result = svc2.decide("whatever", questions)["u"]
    assert result.noul > 0.99  # загружены большие веса, а не случайная инициализация
