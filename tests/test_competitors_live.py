"""Тесты live-конкурента: скоринг кандидатов, движок и провод SystemOne.

Модели — крошечные и собираются локально (без загрузки весов из сети), поэтому
тесты быстрые и воспроизводимые: проверяется математика скоринга и формат
ответа, а не качество модели.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from eval.competitors.serve import (
    LiveEngine,
    create_app,
    score_candidates,
    softmax,
)
from src.sampler.questions import ChoiceSpec, NoulSpec, ScoreSpec


class _OrdTokenizer:
    """Детерминированный байтовый токенизатор: символ -> id (1..255)."""

    pad_token_id = 0

    def __call__(self, text, add_special_tokens=False, **kwargs):
        ids = [(ord(ch) % 255) + 1 for ch in text] or [1]
        return {"input_ids": ids}


class _ConstantLM:
    """Нулевые логиты: все продолжения равновероятны."""

    def __init__(self, vocab: int = 256):
        self.vocab = vocab

    def __call__(self, input_ids, attention_mask=None):
        batch, length = input_ids.shape
        return SimpleNamespace(logits=torch.zeros(batch, length, self.vocab))


class _FavorLM:
    """Высокий логит одному id — для проверки ранжирования кандидатов."""

    def __init__(self, favored_id: int, vocab: int = 256, boost: float = 10.0):
        self.favored_id = favored_id
        self.vocab = vocab
        self.boost = boost

    def __call__(self, input_ids, attention_mask=None):
        batch, length = input_ids.shape
        logits = torch.zeros(batch, length, self.vocab)
        logits[..., self.favored_id] = self.boost
        return SimpleNamespace(logits=logits)


def _tiny_lm():
    from transformers import GPT2Config, GPT2LMHeadModel

    torch.manual_seed(0)
    config = GPT2Config(
        vocab_size=256,
        n_positions=64,
        n_embd=16,
        n_layer=1,
        n_head=1,
    )
    return GPT2LMHeadModel(config).eval()


def _engine() -> LiveEngine:
    return LiveEngine(
        _tiny_lm(), _OrdTokenizer(), model_id="tiny", device="cpu", max_length=64
    )


def _specs() -> dict:
    return {
        "urgent": NoulSpec("Клиент сигнализирует о срочности"),
        "who": ChoiceSpec("Команда?", ("биллинг", "интеграция", "продажи")),
        "heat": ScoreSpec("Накал?", ("низкий", "средний", "высокий")),
    }


def test_softmax_sums_to_one_and_is_monotonic():
    probs = softmax([1.0, 2.0, 3.0])
    assert sum(probs) == pytest.approx(1.0)
    assert probs[0] < probs[1] < probs[2]
    assert softmax([]) == []


def test_score_candidates_constant_model_is_uniform():
    scores = score_candidates(
        _ConstantLM(), _OrdTokenizer(), "n:", ["X", "Y"], device="cpu", max_length=32
    )
    assert scores[0] == pytest.approx(scores[1])
    assert softmax(scores) == pytest.approx([0.5, 0.5])


def test_score_candidates_ranks_favored_continuation():
    favored = (ord("X") % 255) + 1
    scores = score_candidates(
        _FavorLM(favored),
        _OrdTokenizer(),
        "n:",
        ["X", "Y"],
        device="cpu",
        max_length=32,
    )
    assert scores[0] > scores[1]


def test_score_candidates_truncates_long_prompt():
    prompt = "n:" + "z" * 500
    scores = score_candidates(
        _ConstantLM(), _OrdTokenizer(), prompt, ["X", "Y"], device="cpu", max_length=32
    )
    assert len(scores) == 2
    assert all(torch.isfinite(torch.tensor(scores)))


def test_engine_answers_cover_all_question_types():
    answers = _engine().answer("state text", _specs())

    assert set(answers) == {"urgent", "who", "heat"}
    noul = answers["urgent"]
    assert 0.0 <= noul.probability <= 1.0
    assert 0.0 <= noul.confidence <= 1.0

    choice = answers["who"]
    assert choice.choice in _specs()["who"].options
    assert sum(choice.probabilities.values()) == pytest.approx(1.0)

    score = answers["heat"]
    assert score.legend == list(_specs()["heat"].levels)
    assert sum(score.probabilities.values()) == pytest.approx(1.0)


def test_systemone_wire_format_round_trips():
    from fastapi.testclient import TestClient

    app = create_app(_engine())
    client = TestClient(app)
    assert client.get("/health").json()["status"] == "ok"

    payload = {
        "model": "tiny",
        "state": "state text",
        "questions": {
            "urgent": {"type": "noul", "instructions": "Срочно?"},
            "who": {"type": "choice", "instructions": "Кто?", "options": ["a", "b"]},
            "heat": {
                "type": "score",
                "instructions": "Накал?",
                "levels": ["low", "high"],
            },
        },
    }
    response = client.post("/v1/systemone", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "tiny"
    assert set(body["answers"]) == {"urgent", "who", "heat"}
    assert body["answers"]["who"]["type"] == "choice"
    assert body["usage"]["questions"] == 3
