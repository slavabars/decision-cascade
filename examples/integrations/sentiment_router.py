"""Маршрутизация по тональности: тональность + приоритет -> очередь.

Ещё один способ связать решения с действием: определив тональность и срочность,
раскладываем обращения по очередям (VIP-эскалация, обычная линия, авто-ответ).

Запуск:

    .venv/bin/python examples/integrations/sentiment_router.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples._common import (
    UNTRAINED_NOTE,
    build_service,
    primary_value,
    specs_from_api,
)

QUESTIONS = {
    "тональность": {
        "type": "choice",
        "instructions": "Эмоциональная окраска обращения",
        "options": ["позитив", "нейтрально", "недоволен", "злость"],
    },
    "срочно": {"type": "noul", "instructions": "Требуется срочная реакция"},
}

MESSAGES = [
    "Спасибо, всё работает, вы супер!",
    "Как поменять email в профиле?",
    "Верните деньги немедленно, иначе подам в суд!",
]

TONES = ("позитив", "нейтрально", "недоволен", "злость")


def route(text: str, service) -> str:
    decisions = service.decide(text, specs_from_api(QUESTIONS))
    tone = primary_value(decisions, "тональность")
    urgent = float(primary_value(decisions, "срочно") or 0.0)
    if tone == "злость" or urgent >= 0.7:
        return "VIP-эскалация"
    if tone == "недоволен":
        return "старшая линия"
    if tone == "позитив":
        return "запрос отзыва"
    return "обычная линия"


def main() -> None:
    service = build_service()
    for message in MESSAGES:
        print(f"{route(message, service):<16} <- {message[:50]!r}")
    print(f"\n{UNTRAINED_NOTE}")


if __name__ == "__main__":
    main()
