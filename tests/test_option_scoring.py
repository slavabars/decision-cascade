"""Тесты option-scoring головы: билинейная оценка вариантов и перенос набора.

Ключевая проверка — голова обязана *зависеть от состояния*: аддитивная форма
``linear([f(h) ; g(e_i)])`` вырождена (вклад состояния одинаков для всех
вариантов), поэтому регрессионный тест фиксирует именно это свойство.
"""
import pytest
import torch

from api.server.service import DecisionService
from eval.benchmark.core import evaluate_benchmark, train_benchmark
from eval.option_transfer.core import (
    SEEN_OPTIONS,
    generate_examples,
    option_transfer_specs,
)
from src.decision_head.heads import (
    DecisionHeads,
    OptionScoringChoiceHead,
    build_option_embeddings,
)
from src.sampler.questions import (
    ChoiceSpec,
    NoulSpec,
    build_option_prompt,
    questions_signature,
    spec_to_dict,
)


def _identity_head(dim: int, options: tuple[str, ...], embeddings: torch.Tensor):
    """Голова с тождественными проекциями: логит = скалярное произведение."""
    head = OptionScoringChoiceHead(dim, options, embeddings)
    with torch.no_grad():
        head.state_proj.weight.copy_(torch.eye(dim))
        head.option_proj.weight.copy_(torch.eye(dim))
    return head


# --- build_option_prompt / build_option_embeddings ---


def test_build_option_prompt_contains_question_and_option():
    spec = ChoiceSpec("rate the meal", ("poor", "good"))
    prompt = build_option_prompt(spec, "poor")
    assert "rate the meal" in prompt
    assert "poor" in prompt
    # состояние не участвует: вектор варианта переиспользуется для любых состояний
    assert "State:" not in prompt


def test_build_option_prompt_rejects_unknown_option():
    spec = ChoiceSpec("rate the meal", ("poor", "good"))
    with pytest.raises(ValueError):
        build_option_prompt(spec, "excellent")


def test_build_option_embeddings_shape_and_scope():
    spec = ChoiceSpec("who?", ("a", "b", "c"))
    calls: list[list[str]] = []

    def encode(prompts: list[str]) -> torch.Tensor:
        calls.append(list(prompts))
        return torch.randn(len(prompts), 5)

    embeddings = build_option_embeddings(
        {"c": spec, "u": NoulSpec("urgent?")}, encode
    )
    assert set(embeddings) == {"c"}  # Noul не даёт эмбеддингов вариантов
    assert tuple(embeddings["c"].shape) == (3, 5)
    assert len(calls) == 1 and len(calls[0]) == 3


# --- механизм головы ---


def test_option_scoring_logits_shape_and_distribution():
    head = _identity_head(3, ("a", "b", "c"), torch.eye(3))
    logits = head.logits(torch.randn(4, 3))
    assert tuple(logits.shape) == (4, 3)
    res = head(torch.tensor([[1.0, 0.0, 0.0]]))
    assert res.type == "choice"
    assert set(res.probabilities) == {"a", "b", "c"}
    assert abs(sum(res.probabilities.values()) - 1.0) < 1e-6


def test_option_scoring_choice_depends_on_state():
    """Регрессия: аддитивная форма всегда выбирала бы один и тот же вариант."""
    head = _identity_head(3, ("a", "b", "c"), torch.eye(3))
    assert head(torch.tensor([[9.0, 0.0, 0.0]])).choice == "a"
    assert head(torch.tensor([[0.0, 9.0, 0.0]])).choice == "b"
    assert head(torch.tensor([[0.0, 0.0, 9.0]])).choice == "c"


def test_option_scoring_transfers_weights_to_new_option_set():
    """Те же веса, другой набор вариантов (и другой ширины)."""
    dim = 4
    base = _identity_head(dim, ("a", "b", "c"), torch.eye(dim)[:3])
    extended = OptionScoringChoiceHead(
        dim, ("w", "x", "y", "z"), torch.eye(dim)
    )
    extended.load_state_dict(base.state_dict())  # формы state_proj/option_proj совпадают
    res = extended(torch.tensor([[0.0, 0.0, 0.0, 7.0]]))
    assert res.choice == "z"
    assert set(res.probabilities) == {"w", "x", "y", "z"}


def test_option_scoring_rejects_bad_embedding_shape():
    with pytest.raises(ValueError):
        OptionScoringChoiceHead(3, ("a", "b"), torch.zeros(3, 3))


def test_option_scoring_rejects_wrong_feature_dim():
    head = _identity_head(3, ("a", "b"), torch.eye(3)[:2])
    with pytest.raises(ValueError):
        head.logits(torch.randn(2, 5))


# --- DecisionHeads ---


def test_decision_heads_option_scoring_requires_embeddings():
    with pytest.raises(ValueError):
        DecisionHeads(3, {"c": ChoiceSpec("who?", ("a", "b"))}, choice_head="option_scoring")


def test_decision_heads_rejects_unknown_choice_head():
    with pytest.raises(ValueError):
        DecisionHeads(3, {"c": ChoiceSpec("who?", ("a", "b"))}, choice_head="magic")


def test_decision_heads_option_scoring_roundtrip(tmp_path):
    dim = 4
    questions = {"c": ChoiceSpec("who?", ("a", "b", "c"))}
    embeddings = {"c": torch.eye(dim)[:3]}
    heads = DecisionHeads(
        dim, questions, seed=0, choice_head="option_scoring", option_embeddings=embeddings
    )
    path = tmp_path / "heads.pt"
    heads.save_checkpoint(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    assert payload["choice_head"] == "option_scoring"
    # эмбеддинги вариантов не персистятся — иначе перенос был бы невозможен
    assert not any("option_embeddings" in key for key in payload["state_dict"])

    restored = DecisionHeads.load_checkpoint(path, option_embeddings=embeddings)
    assert restored.choice_head == "option_scoring"
    features = torch.randn(1, dim)
    assert heads.heads["c"](features) == restored.heads["c"](features)


def test_load_legacy_checkpoint_defaults_to_linear(tmp_path):
    questions = {"c": ChoiceSpec("who?", ("a", "b"))}
    heads = DecisionHeads(4, questions, seed=0)
    path = tmp_path / "legacy.pt"
    heads.save_checkpoint(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload.pop("choice_head")  # старый формат
    payload["questions"] = {n: spec_to_dict(s) for n, s in questions.items()}
    torch.save(payload, path)
    restored = DecisionHeads.load_checkpoint(path)
    assert restored.choice_head == "linear"


# --- обучение ---


def test_train_benchmark_option_scoring_learns_in_distribution():
    from api.server.service import HashStateEncoder

    encoder = HashStateEncoder(dim=64)
    specs = option_transfer_specs()
    train = generate_examples(SEEN_OPTIONS, 20, 0)
    test = generate_examples(SEEN_OPTIONS, 20, 1)
    heads, _ = train_benchmark(
        specs, train, encoder, train_steps=400, seed=0, choice_head="option_scoring"
    )
    accuracy = evaluate_benchmark(specs, test, encoder, heads)["questions"]["polarity"][
        "accuracy"
    ]
    assert accuracy > 0.9


# --- сервис ---


def test_service_reports_choice_head():
    svc = DecisionService(seed=0, choice_head="option_scoring")
    assert svc.info()["choice_head"] == "option_scoring"


def test_service_rejects_unknown_choice_head():
    with pytest.raises(ValueError):
        DecisionService(seed=0, choice_head="magic")


def test_service_option_scoring_decides():
    svc = DecisionService(seed=0, choice_head="option_scoring")
    result = svc.decide("some state", {"c": ChoiceSpec("who?", ("a", "b"))})
    assert result["c"].type == "choice"
    assert set(result["c"].probabilities) == {"a", "b"}


def test_service_ignores_checkpoint_with_mismatched_choice_head(tmp_path):
    questions = {"c": ChoiceSpec("who?", ("a", "b"))}
    dim = 8
    linear = DecisionHeads(dim, questions, seed=0, choice_head="linear")
    linear.save_checkpoint(tmp_path / f"{questions_signature(questions)}.pt")

    svc = DecisionService(
        dim=dim, seed=0, checkpoint_dir=tmp_path, choice_head="option_scoring"
    )
    result = svc.decide("state", questions)
    assert result["c"].type == "choice"  # не падает на несовпадении режима
    assert svc.info()["heads"][0]["trained"] is False
