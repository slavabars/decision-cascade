"""Модерация контента: токсичность, категория, серьёзность -> действие.

Пример другого набора вопросов (не тикеты): по решениям выбирается действие
``allow`` / ``review`` / ``block``. Политика — обычные пороги, их легко заменить.

Запуск:

    .venv/bin/python examples/integrations/moderation.py
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
    "токсично": {
        "type": "noul",
        "instructions": "Сообщение содержит оскорбления, угрозы или травлю",
    },
    "категория": {
        "type": "choice",
        "instructions": "Основная категория нарушения",
        "options": ["нет", "оскорбление", "угроза", "спам", "18+"],
    },
    "серьёзность": {
        "type": "score",
        "instructions": "Насколько серьёзно нарушение",
        "levels": ["нет", "низкая", "средняя", "высокая", "критическая"],
    },
}

MESSAGES = [
    "Спасибо за помощь, всё получилось!",
    "Ты бесполезный идиот, я тебя найду и уничтожу",
    "КУПИ КРИПТУ СЕЙЧАС, ПЕРЕЙДИ ПО ССЫЛКЕ!!!",
]


def moderate(text: str, service) -> dict:
    decisions = service.decide(text, specs_from_api(QUESTIONS))
    toxic = float(primary_value(decisions, "токсично") or 0.0)
    severity = float(primary_value(decisions, "серьёзность") or 0.0)
    category = primary_value(decisions, "категория")

    if toxic >= 0.8 or severity >= 3.0:
        action = "block"
    elif toxic >= 0.5 or category not in (None, "нет"):
        action = "review"
    else:
        action = "allow"
    return {"action": action, "category": category, "toxic": toxic, "severity": severity}


def main() -> None:
    service = build_service()
    for message in MESSAGES:
        verdict = moderate(message, service)
        print(f"[{verdict['action']:>6}] {message[:50]!r}")
        print(
            f"         категория={verdict['category']} "
            f"токсично={verdict['toxic']:.2f} серьёзность={verdict['severity']:.2f}"
        )
    print(f"\n{UNTRAINED_NOTE}")


if __name__ == "__main__":
    main()
