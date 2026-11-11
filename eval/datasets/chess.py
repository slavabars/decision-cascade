"""Вопросы шахматного демо: когда быстрому просчёту движка нельзя верить.

Датасет генерирует `examples.games.chess.dataset`: состояние — текстовые
признаки позиции (материал, фаза, разброс оценок кандидатов), метка — расходится
ли быстрый поиск с глубоким. Обучение и калибровка идут штатным рецептом
(`training.recipe.train_heads --specs chess --dataset <каталог>`).

Здесь нет зависимости от `python-chess`: модуль только объявляет контракт
вопросов и читает готовые JSONL, поэтому тренировочная часть работает без
опционального extra `games`.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from eval.datasets.text_dataset import spec_from_api, split_examples

CHESS_QUESTIONS: dict[str, dict[str, Any]] = {
    "need_deep": {
        "type": "noul",
        "instructions": (
            "Быстрый просчёт позиции ненадёжен: нужен более глубокий поиск?"
        ),
    },
}

CHESS_SPECS = {name: spec_from_api(data) for name, data in CHESS_QUESTIONS.items()}

SPLIT_NAMES = ("train", "val", "test")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_splits(path: Path, *, seed: int = 0) -> dict[str, list[dict[str, Any]]]:
    """Сплиты из каталога (`train/val/test.jsonl`) или из одного JSONL.

    Каталог — предпочтительный вариант: тогда обучение и калибровка видят ровно
    одно и то же разбиение, а не два независимых.
    """
    path = Path(path)
    if path.is_dir():
        missing = [name for name in SPLIT_NAMES if not (path / f"{name}.jsonl").is_file()]
        if missing:
            raise FileNotFoundError(
                f"{path}: нет сплитов {missing}; запустите examples.games.chess.dataset"
            )
        return {name: _read_jsonl(path / f"{name}.jsonl") for name in SPLIT_NAMES}
    examples = _read_jsonl(path)
    return split_examples(
        examples, val_fraction=0.2, test_fraction=0.2, seed=seed
    )
