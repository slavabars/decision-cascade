"""Русскоязычный набор: тональность текста -> типизированные вопросы.

До него все числа проекта были на английских данных (AG News, RouterBench), а
сам проект русскоязычный: неизвестно, работает ли конвейер на русском и
насколько. Здесь появляется честный ответ.

Датасет — `MonoHime/ru_sentiment_dataset` (агрегат из шести русскоязычных
источников: новости, отзывы, токсичные комментарии). Метки: `0` neutral,
`1` positive, `2` negative; сплиты автора — train/validation, validation и
служит test-срезом.

Лицензия в карточке датасета не указана, поэтому веса, обученные на нём, **не
публикуются** в релизах: числа и команда воспроизведения — да, пак — нет.

Пример:
    python -m eval.datasets.ru_sentiment --out-dir data/ru_sentiment --limit 6000
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from eval.datasets.text_dataset import spec_from_api, write_jsonl

RU_DATASET = "MonoHime/ru_sentiment_dataset"
#: Ревизия закреплена: без неё Hub отдаёт текущий ``main`` и числа не воспроизводятся.
RU_REVISION = "a66a66ee3a858c2b59e056b3fb7dfaf501fc5425"

RU_LABELS = ("neutral", "positive", "negative")

#: Метки датасета -> три вопроса. ``sentiment`` — исходная метка,
#: ``is_positive``/``is_negative`` — её бинарные проекции.
RU_QUESTIONS: dict[str, dict[str, Any]] = {
    "sentiment": {
        "type": "choice",
        "instructions": "Какова тональность текста?",
        "options": list(RU_LABELS),
    },
    "is_positive": {
        "type": "noul",
        "instructions": "Текст выражает положительную тональность?",
    },
    "is_negative": {
        "type": "noul",
        "instructions": "Текст выражает отрицательную тональность?",
    },
}

RU_SPECS = {name: spec_from_api(data) for name, data in RU_QUESTIONS.items()}


def map_example(row_id: str, text: str, label: int) -> dict[str, Any]:
    """Одна строка датасета -> пример с типизированными метками."""
    if not 0 <= label < len(RU_LABELS):
        raise ValueError(f"unknown sentiment label: {label}")
    return {
        "id": row_id,
        "state": text,
        "labels": {
            "sentiment": RU_LABELS[label],
            "is_positive": 1 if label == 1 else 0,
            "is_negative": 1 if label == 2 else 0,
        },
    }


def load_ru_sentiment(
    limit: int | None = None, seed: int = 0
) -> dict[str, list[dict[str, Any]]]:
    """Загружает набор. ``limit`` режет train и пропорционально test.

    Перемешивание с фиксированным ``seed`` обязательно: в исходном файле строки
    сгруппированы по источникам, и «первые N» были бы одним доменом.
    """
    from datasets import load_dataset

    raw = load_dataset(RU_DATASET, revision=RU_REVISION)
    splits = {}
    for split_name, rows in raw.items():
        shuffled = rows.shuffle(seed=seed)
        if limit:
            size = limit if split_name == "train" else max(1, limit // 4)
            shuffled = shuffled.select(range(min(size, len(shuffled))))
        splits["train" if split_name == "train" else "test"] = [
            map_example(
                f"{split_name}-{index:06d}", row["text"], int(row["sentiment"])
            )
            for index, row in enumerate(shuffled)
        ]
    return splits


def label_distribution(examples: list[dict[str, Any]]) -> dict[str, int]:
    """Сколько примеров каждого класса — для отчёта и baseline большинства."""
    counts = {name: 0 for name in RU_LABELS}
    for example in examples:
        counts[example["labels"]["sentiment"]] += 1
    return counts


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Materialize ru_sentiment as typed JSONL"
    )
    parser.add_argument("--out-dir", type=Path, default=Path("data/ru_sentiment"))
    parser.add_argument("--limit", type=int, default=6000, help="0 = весь датасет")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    splits = load_ru_sentiment(args.limit or None, seed=args.seed)
    for split_name, examples in splits.items():
        path = write_jsonl(args.out_dir / f"{split_name}.jsonl", examples)
        print(f"{split_name:<5}: {len(examples):>6} -> {path}")
        print(f"       labels: {label_distribution(examples)}")


if __name__ == "__main__":
    main()
