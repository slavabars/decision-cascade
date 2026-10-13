"""Telegram-бот: сообщение -> решения -> ответ (готовая заготовка).

Без токена запускается dry-run на демо-сообщении (ничего наружу не уходит).
С токеном — long polling через Telegram Bot API; решения считаются in-process,
поэтому отдельный сервер не нужен.

Запуск:

    .venv/bin/python examples/integrations/telegram_bot.py          # dry-run
    TELEGRAM_BOT_TOKEN=... .venv/bin/python examples/integrations/telegram_bot.py --max-updates 5
"""
import argparse
import os
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from api.sdk.agent import route_ticket
from examples._common import build_service, ticket_specs

TELEGRAM_API = "https://api.telegram.org"
DEMO = (
    "Здравствуйте! Вчера сменили тариф, и списания идут дважды. Деньги "
    "списываются, а услуга не подключена. Если сегодня не вернёте — уйду "
    "к конкурентам."
)


class TelegramBot:
    def __init__(self, token: str, service) -> None:
        self._client = httpx.Client(
            base_url=f"{TELEGRAM_API}/bot{token}", timeout=65.0
        )
        self._service = service

    def reply_for(self, text: str) -> str:
        decisions = self._service.decide(text, ticket_specs())
        return route_ticket(decisions).reply

    def run(self, max_updates: int = 0) -> None:
        offset = 0
        handled = 0
        while True:
            resp = self._client.get(
                "/getUpdates", params={"timeout": 30, "offset": offset}
            )
            resp.raise_for_status()
            for update in resp.json().get("result", []):
                offset = update["update_id"] + 1
                message = update.get("message") or {}
                text = message.get("text")
                chat_id = (message.get("chat") or {}).get("id")
                if not text or chat_id is None:
                    continue
                self._client.post(
                    "/sendMessage",
                    json={"chat_id": chat_id, "text": self.reply_for(text)},
                )
                handled += 1
            if max_updates and handled >= max_updates:
                break
        print(f"Обработано сообщений: {handled}")

    def close(self) -> None:
        self._client.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Telegram bot on decision-cascade API")
    parser.add_argument("--token", default=os.environ.get("TELEGRAM_BOT_TOKEN"))
    parser.add_argument("--max-updates", type=int, default=0, help="0 = бесконечно")
    args = parser.parse_args()

    service = build_service()
    if not args.token:
        reply = TelegramBot("dry-run", service).reply_for(DEMO)
        print("[dry-run] нет TELEGRAM_BOT_TOKEN — показываю ответ на демо-сообщение:")
        print(f"  {reply}")
        return

    bot = TelegramBot(args.token, service)
    try:
        bot.run(args.max_updates)
    finally:
        bot.close()


if __name__ == "__main__":
    main()
