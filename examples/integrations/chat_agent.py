"""Пример интеграции decision-cascade API в чат-агента (SDK + каскад).

Показывает, как по типизированным решениям маршрутизировать обращение: выбрать
команду, приоритет, эскалацию и тон ответа. Если включён каскад на стороне
сервера (``DECISION_BACKEND=cascade``), низкоуверенные обращения он эскалирует
сам; здесь же показан и клиентский фолбэк: при ``--fallback`` решающий агент
переспрашивает внешний сервис, когда ``эскалация`` срабатывает.

Запуск (сервер должен быть поднят, см. README):

    .venv/bin/python -m uvicorn api.server.main:app --port 8000
    .venv/bin/python examples/integrations/chat_agent.py
    .venv/bin/python examples/integrations/chat_agent.py --fallback http://remote:8080
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from api.sdk.agent import route_ticket
from api.sdk.client import DecisionClient
from eval.datasets.text_dataset import QUESTIONS

STATE = (
    "Здравствуйте! Вчера сменили тариф, и теперь списания идут дважды. "
    "Деньги списываются, а услуга не подключена. Я уже писал трижды, ответа нет. "
    "Если сегодня не вернёте деньги, я оставлю негативный отзыв и уйду к "
    "конкурентам. Верните, пожалуйста, средства и разберитесь с биллингом."
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Routing a ticket with decision-cascade API")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument(
        "--fallback",
        default=None,
        help="URL внешнего SystemOne-сервиса для клиентского каскада",
    )
    args = parser.parse_args()

    with DecisionClient(base_url=args.base_url) as client:
        payload = client.decisions(STATE, QUESTIONS)
        action = route_ticket(payload["decisions"])

        print(f"Обращение: {STATE[:70]}...")
        print(f"  команда:   {action.team}")
        print(f"  приоритет: {action.priority}  (эскалация: {action.escalate})")
        print(f"  тон:       {action.tone}")
        print(f"  ответ:     {action.reply}")
        print("  почему:")
        for reason in action.rationale:
            print(f"    - {reason}")

        if action.escalate and args.fallback:
            print(f"  каскад -> внешний сервис {args.fallback}")
            with DecisionClient(base_url=args.fallback) as remote:
                remote_payload = remote.decisions(STATE, QUESTIONS)
            remote_action = route_ticket(remote_payload["decisions"])
            print(f"    внешний ответ: команда={remote_action.team}, "
                  f"приоритет={remote_action.priority}")


if __name__ == "__main__":
    main()
