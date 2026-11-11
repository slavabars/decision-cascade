"""Контракт вопроса для головы лабиринта."""
from __future__ import annotations

from typing import Any

from eval.datasets.text_dataset import spec_from_api

MAZE_QUESTIONS: dict[str, dict[str, Any]] = {
    "need_deep": {
        "type": "noul",
        "instructions": (
            "Жадный шаг к ближайшей точке приведёт к потере жизни в ближайшие "
            "12 ходов?"
        ),
    },
}

MAZE_SPECS = {name: spec_from_api(data) for name, data in MAZE_QUESTIONS.items()}
