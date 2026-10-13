"""Публичный датасет AG News -> типизированные вопросы (честная оценка).

В отличие от синтетического генератора (``text_dataset``) метки здесь реальные
и не линейно отделимы, поэтому accuracy < 1.0 — это проверка обобщения, а не
конвейера. Маппинг на ``noul``/``choice``/``score`` детерминированный и описан
рядом с ``PUBLIC_QUESTIONS``.

Примеры:

    python -m eval.datasets.public_dataset --out-dir data/ag_news --limit 4000
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from eval.datasets.text_dataset import spec_from_api, write_jsonl

AG_NEWS_LABELS = ("World", "Sports", "Business", "Sci/Tech")

# Реальные метки AG News -> три типизированных вопроса. ``topic`` — настоящая
# метка; ``business`` — бинарная проекция; ``tech_level`` — порядковая проекция
# (Sci/Tech -> high, Business -> medium, остальное -> low).
PUBLIC_QUESTIONS: dict[str, dict[str, Any]] = {
    "topic": {
        "type": "choice",
        "instructions": "Which section does this news belong to?",
        "options": list(AG_NEWS_LABELS),
    },
    "business": {
        "type": "noul",
        "instructions": "Is this business or markets news?",
    },
    "tech_level": {
        "type": "score",
        "instructions": "How technology-focused is this story?",
        "levels": ["low", "medium", "high"],
    },
}

PUBLIC_SPECS = {name: spec_from_api(data) for name, data in PUBLIC_QUESTIONS.items()}


def _tech_level(label: int) -> int:
    if label == 3:  # Sci/Tech
        return 2
    if label == 2:  # Business
        return 1
    return 0


def map_example(row_id: str, text: str, label: int) -> dict[str, Any]:
    """Одна строка AG News -> пример с типизированными метками."""
    if not 0 <= label < len(AG_NEWS_LABELS):
        raise ValueError(f"unknown AG News label: {label}")
    return {
        "id": row_id,
        "state": text,
        "labels": {
            "topic": AG_NEWS_LABELS[label],
            "business": 1 if label == 2 else 0,
            "tech_level": _tech_level(label),
        },
    }


def load_ag_news(limit: int | None = None) -> dict[str, list[dict[str, Any]]]:
    """Загружает AG News (train/test). ``limit`` режет train и пропорционально test."""
    from datasets import load_dataset

    raw = load_dataset("fancyzhx/ag_news")
    splits = {}
    for split_name, rows in raw.items():
        splits[split_name] = [
            map_example(f"{split_name}-{i:06d}", row["text"], int(row["label"]))
            for i, row in enumerate(rows)
        ]
    if limit:
        train = splits.get("train", [])[:limit]
        test = splits.get("test", [])[: max(1, limit // 4)]
        splits = {"train": train, "test": test}
    return splits


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Materialize AG News as typed JSONL")
    parser.add_argument("--out-dir", type=Path, default=Path("data/ag_news"))
    parser.add_argument("--limit", type=int, default=4000, help="0 = весь датасет")
    args = parser.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    splits = load_ag_news(args.limit or None)
    for split_name, examples in splits.items():
        path = write_jsonl(args.out_dir / f"{split_name}.jsonl", examples)
        print(f"{split_name:<5}: {len(examples):>6} -> {path}")


if __name__ == "__main__":
    main()
