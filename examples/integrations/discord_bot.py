"""Discord: решение -> ответ в канал (REST, без websocket-зависимостей).

Без ``DISCORD_BOT_TOKEN`` и ``DISCORD_CHANNEL_ID`` — dry-run. С ними отправляет
ответ на демо-сообщение в канал через Discord REST API. В реальном боте этот же
``reply_for`` вызывается из обработчика события MESSAGE_CREATE.

Запуск:

    .venv/bin/python examples/integrations/discord_bot.py            # dry-run
    DISCORD_BOT_TOKEN=... DISCORD_CHANNEL_ID=... \\
        .venv/bin/python examples/integrations/discord_bot.py --send
"""
import argparse
import os
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from api.sdk.agent import route_ticket
from examples._common import build_service, ticket_specs

DISCORD_API = "https://discord.com/api/v10"
DEMO = "Списание прошло, а услуга не подключена. Верните деньги!"


class DiscordNotifier:
    def __init__(self, token: str, channel_id: str, service) -> None:
        self._client = httpx.Client(
            base_url=DISCORD_API,
            headers={"Authorization": f"Bot {token}"},
            timeout=30.0,
        )
        self._channel_id = channel_id
        self._service = service

    def reply_for(self, text: str) -> str:
        decisions = self._service.decide(text, ticket_specs())
        return route_ticket(decisions).reply

    def send(self, content: str) -> None:
        resp = self._client.post(
            f"/channels/{self._channel_id}/messages", json={"content": content}
        )
        resp.raise_for_status()

    def close(self) -> None:
        self._client.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Discord notifier on Decision API")
    parser.add_argument("--token", default=os.environ.get("DISCORD_BOT_TOKEN"))
    parser.add_argument("--channel", default=os.environ.get("DISCORD_CHANNEL_ID"))
    parser.add_argument("--send", action="store_true", help="реально отправить")
    args = parser.parse_args()

    service = build_service()
    if not args.token or not args.channel:
        reply = DiscordNotifier("dry-run", "0", service).reply_for(DEMO)
        print("[dry-run] нет DISCORD_BOT_TOKEN/DISCORD_CHANNEL_ID:")
        print(f"  {reply}")
        return

    notifier = DiscordNotifier(args.token, args.channel, service)
    try:
        reply = notifier.reply_for(DEMO)
        print(f"Ответ: {reply}")
        if args.send:
            notifier.send(reply)
            print("Отправлено в канал.")
        else:
            print("Добавьте --send, чтобы реально отправить.")
    finally:
        notifier.close()


if __name__ == "__main__":
    main()
