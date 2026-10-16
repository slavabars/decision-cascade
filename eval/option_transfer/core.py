"""Zero-shot перенос вариантов выбора.

Замеряет, что умеет option-scoring голова, а что нет. Задача синтетическая и
полностью офлайн: состояние — короткая фраза о еде, полярность задана словами из
словаря, **не пересекающегося с вариантами** (иначе голова могла бы угадать ответ
по совпадению слова, а не по семантике).

Проверяются четыре вещи:

1. ``seen`` — обычная точность на отложенных примерах (варианты из обучения).
2. ``zero_shot`` — голова обучена на трёх вариантах, а на инференсе ей подменяют
   набор из двух **невиданных** вариантов, те же веса. Retrained-эталон — та же
   голова, обученная на этом наборе с нуля (верхняя граница без переноса).
3. ``distractors`` — набор из всех пяти вариантов: выбираются ли невиданные
   варианты, когда рядом есть знакомые.
4. ``new_words`` — набор из двух **новых слов**, которых не было даже в вопросе.

Линейная голова для 2–4 неприменима: её выходной слой имеет фиксированную ширину
(число вариантов на момент обучения), поэтому перенос невозможен в принципе —
это и есть измеряемое преимущество option-scoring.

Метрики считаются по нескольким seed'ам (mean ± std): перенос чувствителен к
инициализации и геометрии энкодера, и одиночный прогон вводил бы в заблуждение.
"""
from __future__ import annotations

import random
import statistics
from typing import Any

import torch

from eval.benchmark.core import TextEncoder, evaluate_benchmark, train_benchmark
from src.decision_head.heads import OptionScoringChoiceHead
from src.sampler.questions import (
    ChoiceSpec,
    QuestionSpec,
    build_option_prompt,
    build_prompt,
)

INSTRUCTIONS = "rate the meal"

#: Вопрос со всеми вариантами — используется как «набор с отвлекающими».
ALL_OPTIONS = ("dreadful", "poor", "neutral", "good", "excellent")
#: Варианты, представленные в обучении.
SEEN_OPTIONS = ("poor", "neutral", "good")
#: Невиданные варианты, семантически близкие к обученным (полярность по краям).
UNSEEN_OPTIONS = ("dreadful", "excellent")
#: Новые слова, которых не было даже в вопросе (проверка границы переноса).
NEW_OPTIONS = ("terrible", "flawless")

QUESTION = "polarity"

# Словарь состояний. Намеренно не пересекается с вариантами.
_VOCAB: dict[str, tuple[str, ...]] = {
    "dreadful": ("horrendous", "appalling", "atrocious"),
    "poor": ("disappointing", "mediocre", "underwhelming"),
    "neutral": ("average", "ordinary", "unremarkable"),
    "good": ("pleasant", "satisfying", "solid"),
    "excellent": ("outstanding", "superb", "brilliant"),
    # только для NEW_OPTIONS — появляются лишь в новом наборе
    "terrible": ("dismal", "wretched", "hideous"),
    "flawless": ("perfect", "immaculate", "exquisite"),
}


def option_transfer_specs(
    options: tuple[str, ...] = ALL_OPTIONS,
) -> dict[str, QuestionSpec]:
    """Вопрос полярности с заданным набором вариантов."""
    return {QUESTION: ChoiceSpec(INSTRUCTIONS, options)}


def generate_examples(
    labels: tuple[str, ...], n_per_label: int, seed: int
) -> list[dict[str, Any]]:
    """Примеры ``{"state": ..., "labels": {"polarity": <опция>}}``."""
    rng = random.Random(seed)
    examples: list[dict[str, Any]] = []
    for _ in range(n_per_label):
        for label in labels:
            vocab = _VOCAB[label]
            first = rng.choice(vocab)
            pool = [word for word in vocab if word != first] or list(vocab)
            second = rng.choice(pool)
            examples.append(
                {
                    "state": f"The meal was {first} and the service was {second}.",
                    "labels": {QUESTION: label},
                }
            )
    rng.shuffle(examples)
    return examples


def run_option_transfer(
    encoder: TextEncoder,
    *,
    train_steps: int = 800,
    lr: float = 1e-3,
    seeds: tuple[int, ...] = (0, 1, 2),
    n_per_label: int = 40,
) -> dict[str, Any]:
    """Перенос вариантов по нескольким seed'ам; возвращает агрегированный отчёт."""
    per_seed = [
        _run_seed(encoder, seed, train_steps=train_steps, lr=lr, n_per_label=n_per_label)
        for seed in seeds
    ]
    return {
        "encoder": type(encoder).__name__,
        "seeds": list(seeds),
        "seen_options": list(SEEN_OPTIONS),
        "unseen_options": list(UNSEEN_OPTIONS),
        "new_options": list(NEW_OPTIONS),
        "summary": _aggregate(per_seed),
        "per_seed": per_seed,
    }


def _run_seed(
    encoder: TextEncoder, seed: int, *, train_steps: int, lr: float, n_per_label: int
) -> dict[str, Any]:
    seen_spec = option_transfer_specs(SEEN_OPTIONS)
    train_seen = generate_examples(SEEN_OPTIONS, n_per_label, seed)
    seen_test = generate_examples(SEEN_OPTIONS, n_per_label, seed + 1)
    unseen_test = generate_examples(UNSEEN_OPTIONS, n_per_label, seed + 1)
    new_test = generate_examples(NEW_OPTIONS, n_per_label, seed + 1)

    scoring, _ = train_benchmark(
        seen_spec, train_seen, encoder,
        train_steps=train_steps, lr=lr, seed=seed, choice_head="option_scoring",
    )
    linear, _ = train_benchmark(
        seen_spec, train_seen, encoder,
        train_steps=train_steps, lr=lr, seed=seed, choice_head="linear",
    )
    dim = int(encoder.encode([train_seen[0]["state"]]).shape[-1])
    # Фактическая ширина выхода линейной головы — она закреплена на обучении и
    # потому не позволяет подменить набор вариантов (измеряется, а не предполагается).
    linear_width = int(linear.heads[QUESTION].logits(torch.zeros(1, dim)).shape[-1])

    return {
        "seed": seed,
        "seen": {
            "option_scoring": evaluate_benchmark(
                seen_spec, seen_test, encoder, scoring
            )["questions"][QUESTION]["accuracy"],
            "linear": evaluate_benchmark(seen_spec, seen_test, encoder, linear)[
                "questions"
            ][QUESTION]["accuracy"],
        },
        "zero_shot": _zero_shot(
            encoder, dim, scoring, UNSEEN_OPTIONS, unseen_test,
            baseline=_retrained_linear(
                encoder, UNSEEN_OPTIONS, seed, train_steps, lr, n_per_label,
                unseen_test,
            ),
        ),
        "distractors": _zero_shot(
            encoder, dim, scoring, ALL_OPTIONS, unseen_test
        ),
        "new_words": _zero_shot(
            encoder, dim, scoring, NEW_OPTIONS, new_test,
            baseline=_retrained_linear(
                encoder, NEW_OPTIONS, seed, train_steps, lr, n_per_label, new_test,
            ),
        ),
        "linear_option_set_width": linear_width,
    }


def _retrained_linear(
    encoder: TextEncoder,
    options: tuple[str, ...],
    seed: int,
    train_steps: int,
    lr: float,
    n_per_label: int,
    test_examples: list[dict[str, Any]],
) -> float:
    """Эталон «обучить с нуля на этом наборе» — верхняя граница без переноса."""
    spec = option_transfer_specs(options)
    train = generate_examples(options, n_per_label, seed)
    heads, _ = train_benchmark(
        spec, train, encoder, train_steps=train_steps, lr=lr, seed=seed,
        choice_head="linear",
    )
    return evaluate_benchmark(spec, test_examples, encoder, heads)["questions"][
        QUESTION
    ]["accuracy"]


def _zero_shot(
    encoder: TextEncoder,
    dim: int,
    trained_heads,
    options: tuple[str, ...],
    test_examples: list[dict[str, Any]],
    baseline: float | None = None,
) -> dict[str, Any]:
    """Точность option-scoring головы на подменённом наборе вариантов (те же веса).

    ``baseline`` — точность головы, обученной на этом наборе с нуля (верхняя
    граница без переноса).
    """
    spec = option_transfer_specs(options)
    head = _rebuilt_head(encoder, dim, options, trained_heads.heads[QUESTION])
    features = encoder.encode(
        [build_prompt(e["state"], spec[QUESTION]) for e in test_examples]
    )
    with torch.no_grad():
        logits = head.logits(features)
    preds = [head.options[i] for i in logits.argmax(-1).tolist()]
    labels = [e["labels"][QUESTION] for e in test_examples]
    accuracy = sum(p == lab for p, lab in zip(preds, labels)) / len(labels)
    result: dict[str, Any] = {
        "n_options": len(options),
        "chance": 1.0 / len(options),
        "option_scoring": accuracy,
    }
    if baseline is not None:
        result["linear_retrained"] = baseline
    return result


def _rebuilt_head(
    encoder: TextEncoder, dim: int, options: tuple[str, ...], trained_head
) -> OptionScoringChoiceHead:
    """Новая option-scoring голова: те же обученные веса, другой набор вариантов.

    ``option_embeddings`` — non-persistent буфер (в ``state_dict`` не входит),
    поэтому в новую голову копируются только обучаемые параметры ``state_proj`` и
    ``option_proj``; их формы не зависят от числа вариантов.
    """
    embeddings = encoder.encode(
        [build_option_prompt(ChoiceSpec(INSTRUCTIONS, options), opt) for opt in options]
    )
    head = OptionScoringChoiceHead(dim, options, embeddings)
    head.load_state_dict(trained_head.state_dict())
    return head


def _aggregate(per_seed: list[dict[str, Any]]) -> dict[str, Any]:
    """mean ± std по seed'ам для скалярных метрик."""
    keys: dict[str, list[float]] = {}
    for record in per_seed:
        for key, value in _flatten(record).items():
            keys.setdefault(key, []).append(value)
    return {
        key: {
            "mean": statistics.mean(values),
            "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
        }
        for key, values in keys.items()
    }


def _flatten(record: dict[str, Any], prefix: str = "") -> dict[str, float]:
    flat: dict[str, float] = {}
    for key, value in record.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(_flatten(value, f"{name}."))
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            flat[name] = float(value)
    return flat
