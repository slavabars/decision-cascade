"""Письмо -> структурированный триаж + черновик ответа.

Берёт тему и тело письма, прогоняет через решения и печатает готовую карточку:
команда, приоритет, эскалация, тон и черновик ответа.

Запуск:

    .venv/bin/python examples/integrations/email_triage.py
    .venv/bin/python examples/integrations/email_triage.py \\
        --subject "Двойное списание" --body "Верните деньги, иначе уйду"
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from api.sdk.agent import route_ticket
from examples._common import build_service, primary_value, ticket_specs

DEMO_SUBJECT = "Двойное списание по тарифу"
DEMO_BODY = (
    "Здравствуйте! Вчера сменили тариф, и теперь списания идут дважды. "
    "Деньги уходят, услуга не подключена. Я уже писал трижды, ответа нет. "
    "Если сегодня не вернёте средства — оставлю негативный отзыв."
)


def triage(subject: str, body: str) -> dict:
    state = f"{subject}\n\n{body}".strip()
    service = build_service()
    decisions = service.decide(state, ticket_specs())
    action = route_ticket(decisions)
    return {
        "команда": action.team,
        "приоритет": action.priority,
        "эскалация": action.escalate,
        "тон": action.tone,
        "накал": primary_value(decisions, "накал"),
        "риск_оттока": primary_value(decisions, "удержание"),
        "черновик": action.reply,
        "обоснование": action.rationale,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Email triage with decisions")
    parser.add_argument("--subject", default=DEMO_SUBJECT)
    parser.add_argument("--body", default=DEMO_BODY)
    args = parser.parse_args()

    card = triage(args.subject, args.body)
    print(f"Тема: {args.subject}\n")
    for key in ("команда", "приоритет", "эскалация", "тон"):
        print(f"{key:<12}: {card[key]}")
    print(f"{'накал':<12}: {float(card['накал'] or 0.0):.2f}")
    print(f"{'риск_оттока':<12}: {float(card['риск_оттока'] or 0.0):.2f}")
    print(f"\nЧерновик ответа:\n  {card['черновик']}")
    print("\nОбоснование:")
    for reason in card["обоснование"]:
        print(f"  - {reason}")


if __name__ == "__main__":
    main()
