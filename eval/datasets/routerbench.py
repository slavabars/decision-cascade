"""Публичный датасет RouterBench -> типизированные вопросы (честная оценка).

RouterBench (Hu et al., 2024, arXiv:2403.12031) — публичный бенчмарк
маршрутизации между LLM: ~36k промптов из стандартных наборов (MMLU, GSM-8K,
HellaSwag, MBPP, ...), для каждого — оценка качества (0..1) и стоимость ответа
11 разных моделей. Это не классификация тем, а настоящая decision-задача:
«стоит ли эскалировать запрос на более сильную (и дорогую) модель». Поэтому
датасет подходит для честной доменной калибровки каскада.

Промпты берутся как есть, а метки выводятся детерминированно из оценок трёх
моделей-уровней (по стоимости):

- ``cheap``  — ``mistralai/mistral-7b-chat`` (самая дешёвая);
- ``mid``    — ``mistralai/mixtral-8x7b-chat``;
- ``strong`` — ``gpt-4-1106-preview`` (самая сильная и дорогая).

Три вопроса покрывают все типы решений и ядро каскада:

- ``escalate`` (noul): дешёвая модель не решит задачу полностью
  (``cheap < 1``) -> нужно эскалировать. Это и есть триггер каскада.
- ``route``    (choice): какой уровень должен отвечать —
  ``cheap``/``mid``/``strong`` (максимум качества, при равенстве — дешевле).
- ``benefit``  (score): насколько сильная модель улучшит ответ над дешёвой
  (``strong - cheap``): ``none``/``some``/``large``.

Метки реальные, промпты разнородные, accuracy < 1.0 — это проверка обобщения,
а не конвейера. В датасете нет готового train/test, поэтому разбиение
детерминированное (по ``seed``) и фиксируется при материализации.

Примеры:

    python -m eval.datasets.routerbench --out-dir data/routerbench --limit 4000
    python -m eval.report.run --specs routerbench \\
        --encoder openai-community/gpt2 --dim 768 --device cuda \\
        --limit 4000 --out-dir data/report_routerbench
"""
from __future__ import annotations

import argparse
import ast
import math
from pathlib import Path
from random import Random
from typing import Any

from eval.datasets.text_dataset import spec_from_api, write_jsonl

ROUTERBENCH_REPO = "withmartian/routerbench"
# Зафиксированная ревизия: воспроизводимость важнее «свежести».
ROUTERBENCH_REVISION = "784021482c3f320c6619ed4b3bb3b41a21424fcb"
ROUTERBENCH_FILES = {
    "0shot": "routerbench_0shot.pkl",
    "5shot": "routerbench_5shot.pkl",
}

CHEAP_MODEL = "mistralai/mistral-7b-chat"
MID_MODEL = "mistralai/mixtral-8x7b-chat"
STRONG_MODEL = "gpt-4-1106-preview"

# Уровни отсортированы по стоимости: при равенстве качества выигрывает дешёвый.
ROUTER_TIERS = ("cheap", "mid", "strong")
ROUTER_MODELS = {
    "cheap": CHEAP_MODEL,
    "mid": MID_MODEL,
    "strong": STRONG_MODEL,
}

# ``escalate`` — настоящая метка RouterBench (ошибка дешёвой модели); ``route`` и
# ``benefit`` — детерминированные проекции оценок на выбор уровня и на выигрыш
# от эскалации. Правила описаны рядом с ``_route_label``/``_benefit_label``.
ROUTERBENCH_QUESTIONS: dict[str, dict[str, Any]] = {
    "escalate": {
        "type": "noul",
        "instructions": (
            "The cheapest model will not fully solve this task; "
            "escalate it to a stronger model."
        ),
    },
    "route": {
        "type": "choice",
        "instructions": "Which model tier should answer this task?",
        "options": list(ROUTER_TIERS),
    },
    "benefit": {
        "type": "score",
        "instructions": (
            "How much would the strongest model improve over the cheapest?"
        ),
        "levels": ["none", "some", "large"],
    },
}

ROUTERBENCH_SPECS = {
    name: spec_from_api(data) for name, data in ROUTERBENCH_QUESTIONS.items()
}


def parse_prompt(prompt: Any) -> str:
    """Промпт RouterBench -> текст состояния.

    В датасете промпт хранится как ``str``-представление списка сообщений
    (строк или словарей с ``content``). Склеиваем содержимое в один текст; при
    неожиданном формате возвращаем исходную строку, не теряя данные.
    """
    if isinstance(prompt, str):
        try:
            messages: Any = ast.literal_eval(prompt)
        except (ValueError, SyntaxError):
            return prompt
    else:
        messages = prompt
    if not isinstance(messages, (list, tuple)):
        return str(messages)
    parts: list[str] = []
    for message in messages:
        if isinstance(message, dict):
            parts.append(str(message.get("content", "")))
        else:
            parts.append(str(message))
    return "\n\n".join(part for part in parts if part)


def _coerce_score(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if math.isnan(score) else score


def _row_scores(row: dict[str, Any]) -> dict[str, float]:
    return {tier: _coerce_score(row[model]) for tier, model in ROUTER_MODELS.items()}


def _escalate_label(cheap_score: float) -> int:
    """1 — дешёвая модель не решила задачу полностью (нужна эскалация)."""
    return 1 if cheap_score < 1.0 else 0


def _route_label(scores: dict[str, float]) -> str:
    """Уровень с максимальным качеством; при равенстве — самый дешёвый."""
    best = ROUTER_TIERS[0]
    for tier in ROUTER_TIERS[1:]:
        if scores[tier] > scores[best]:
            best = tier
    return best


def _benefit_label(gain: float) -> int:
    """Выигрыш сильной модели над дешёвой: none / some / large."""
    if gain <= 0.0:
        return 0
    if gain <= 0.5:
        return 1
    return 2


def map_example(
    row_id: str,
    prompt: Any,
    scores: dict[str, float],
    *,
    eval_name: str | None = None,
) -> dict[str, Any]:
    """Одна строка RouterBench -> пример с типизированными метками."""
    cheap = scores["cheap"]
    strong = scores["strong"]
    return {
        "id": row_id,
        "state": parse_prompt(prompt),
        "labels": {
            "escalate": _escalate_label(cheap),
            "route": _route_label(scores),
            "benefit": _benefit_label(strong - cheap),
        },
        "metadata": {"eval_name": eval_name, "scores": dict(scores)},
    }


def _read_frame(shot: str):
    """Читает pickle-таблицу RouterBench из кэша Hugging Face."""
    if shot not in ROUTERBENCH_FILES:
        raise ValueError(
            f"unknown shot: {shot!r} (expected {sorted(ROUTERBENCH_FILES)})"
        )
    import pandas as pd
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        ROUTERBENCH_REPO,
        ROUTERBENCH_FILES[shot],
        repo_type="dataset",
        revision=ROUTERBENCH_REVISION,
    )
    return pd.read_pickle(path)


def load_routerbench(
    limit: int | None = None,
    *,
    shot: str = "0shot",
    test_fraction: float = 0.25,
    seed: int = 0,
) -> dict[str, list[dict[str, Any]]]:
    """Загружает RouterBench и делит на train/test (готового сплита нет).

    ``limit`` режет train и пропорционально test; и разбиение, и подвыборка
    детерминированы по ``seed`` и берутся из перемешанного порядка (то есть
    случайны, а не срезом по алфавиту ``eval_name``).
    """
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be in (0.0, 1.0)")
    if shot not in ROUTERBENCH_FILES:
        raise ValueError(
            f"unknown shot: {shot!r} (expected {sorted(ROUTERBENCH_FILES)})"
        )
    frame = _read_frame(shot)
    rows = frame.to_dict("records")
    examples = [
        map_example(
            f"routerbench-{shot}-{index:06d}",
            row.get("prompt"),
            _row_scores(row),
            eval_name=row.get("eval_name"),
        )
        for index, row in enumerate(rows)
    ]

    order = list(range(len(examples)))
    Random(seed).shuffle(order)
    n_test = max(1, round(len(examples) * test_fraction)) if examples else 0
    test_order = order[:n_test]
    train_order = order[n_test:]
    if limit:
        train_order = train_order[:limit]
        test_order = test_order[: max(1, limit // 4)]
    train = [examples[i] for i in train_order]
    test = [examples[i] for i in test_order]
    return {"train": train, "test": test}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Materialize RouterBench as typed JSONL"
    )
    parser.add_argument("--out-dir", type=Path, default=Path("data/routerbench"))
    parser.add_argument("--limit", type=int, default=4000, help="0 = весь датасет")
    parser.add_argument("--shot", choices=sorted(ROUTERBENCH_FILES), default="0shot")
    parser.add_argument("--test-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    splits = load_routerbench(
        args.limit or None,
        shot=args.shot,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )
    for split_name, examples in splits.items():
        path = write_jsonl(args.out_dir / f"{split_name}.jsonl", examples)
        print(f"{split_name:<5}: {len(examples):>6} -> {path}")


if __name__ == "__main__":
    main()
