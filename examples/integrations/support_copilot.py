"""Copilot первой линии: решения -> карточка агента поддержки.

Расширяет ``route_ticket`` до готового рабочего места: маршрут, приоритет, тон,
предлагаемый макрос ответа и следующие шаги. Логику легко переложить на свою
политику (``RoutingPolicy``) и свои макросы.

Запуск:

    .venv/bin/python examples/integrations/support_copilot.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from api.sdk.agent import route_ticket
from examples._common import build_service, primary_value, ticket_specs

DEMO = (
    "Не работает интеграция по API: наши вебхуки перестали приниматься, "
    "ключи выдают invalid key. У нас простой, каждый час теряем деньги. "
    "Подключите старшего инженера, сами не справимся."
)

MACROS = {
    "биллинг": "Проверю списания и вернусь с точной суммой в течение часа.",
    "интеграция": "Пришлите, пожалуйста, ID запроса и время последнего вызова.",
    "продажи": "Расскажу про подходящий тариф и условия перехода.",
    "общая поддержка": "Опишите, пожалуйста, шаги, после которых возникла ошибка.",
}


def build_card(text: str) -> dict:
    service = build_service()
    decisions = service.decide(text, ticket_specs())
    action = route_ticket(decisions)
    next_steps = []
    if action.escalate:
        next_steps.append("Передать старшему инженеру")
    if float(primary_value(decisions, "срочность") or 0.0) >= 0.5:
        next_steps.append("Отметить как business-critical")
    if float(primary_value(decisions, "удержание") or 0.0) >= 3.0:
        next_steps.append("Подключить аккаунт-менеджера (риск оттока)")
    next_steps.append(f"Ответить в тоне «{action.tone}»")
    return {
        "route": action,
        "macro": MACROS.get(action.team, MACROS["общая поддержка"]),
        "next_steps": next_steps,
    }


def main() -> None:
    card = build_card(DEMO)
    route = card["route"]
    print(f"Маршрут: {route.team} | приоритет: {route.priority} | "
          f"эскалация: {route.escalate} | тон: {route.tone}")
    print(f"\nМакрос: {card['macro']}")
    print("\nСледующие шаги:")
    for step in card["next_steps"]:
        print(f"  - {step}")
    print(f"\nЧерновик ответа:\n  {route.reply}")


if __name__ == "__main__":
    main()
