"""Минимум по мультимодальности: ``HashVisionEncoder`` — офлайн и детерминированно.

Показывает, как сервис объявляет модальности и как ``media_weight`` влияет на
ответ. Не требует ни моделей, ни сети — годится для тестов и CI.

Запуск:

    .venv/bin/python examples/vision/vision_offline.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples._common import build_service, primary_value, specs_from_api

QUESTIONS = {
    "важно": {"type": "noul", "instructions": "Вложение важно для решения"},
}
STATE = "Текст тикета"


def main() -> None:
    service = build_service(vision=True)
    decisions = service.decide(
        STATE,
        specs_from_api(QUESTIONS),
        images=["some-bytes-of-image"],
        media_kwargs={"media_weight": 0.5},
    )
    print(f"Модальности: {sorted(service.modalities)}")

    for weight in (0.0, 0.5, 1.0):
        result = service.decide(
            STATE,
            specs_from_api(QUESTIONS),
            images=["some-bytes-of-image"],
            media_kwargs={"media_weight": weight},
        )
        value = primary_value(result, "важно")
        print(f"media_weight={weight:.1f} -> важно={float(value or 0.0):.3f}")

    print(f"reference (0.5): {float(primary_value(decisions, 'важно') or 0.0):.3f}")


if __name__ == "__main__":
    main()
