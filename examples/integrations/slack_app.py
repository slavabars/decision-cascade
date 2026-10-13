"""Slack: слэш-команда ``/triage`` -> решения -> ответ (FastAPI-заготовка).

Slack присылает слэш-команду формой ``application/x-www-form-urlencoded``; чтобы
не тащить лишние зависимости, тело разбирается стандартным ``urllib.parse``.
Без ``--serve`` запускается офлайн-демонстрация через ``TestClient``.

Запуск:

    .venv/bin/python examples/integrations/slack_app.py           # демо
    .venv/bin/python examples/integrations/slack_app.py --serve   # uvicorn :8000
    # затем URL команды в Slack: https://<host>/slack/triage
"""
import argparse
import sys
from pathlib import Path
from urllib.parse import parse_qs

from fastapi import FastAPI, Request

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from api.sdk.agent import route_ticket
from examples._common import build_service, ticket_specs

app = FastAPI(title="Decision Slack app")
_service = build_service()


@app.post("/slack/triage")
async def triage(request: Request) -> dict:
    """Текст команды -> решения -> ответ для Slack."""
    raw = (await request.body()).decode("utf-8")
    text = (parse_qs(raw).get("text") or [""])[0].strip()
    decisions = _service.decide(text or "пустое обращение", ticket_specs())
    action = route_ticket(decisions)
    summary = (
        f"*{action.team}* · приоритет: {action.priority} · тон: {action.tone}"
    )
    return {
        "response_type": "in_channel",
        "text": action.reply,
        "attachments": [{"text": summary}],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Slack slash-command app")
    parser.add_argument("--serve", action="store_true", help="запустить uvicorn")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    if args.serve:
        import uvicorn

        uvicorn.run(app, port=args.port)
        return

    from fastapi.testclient import TestClient

    client = TestClient(app)
    resp = client.post(
        "/slack/triage",
        content="text=Деньги списались дважды, верните и разберитесь",
        headers={"content-type": "application/x-www-form-urlencoded"},
    )
    print(f"[демо] HTTP {resp.status_code}")
    print(resp.json())


if __name__ == "__main__":
    main()
