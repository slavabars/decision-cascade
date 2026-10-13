"""Встроить решения в существующее FastAPI-приложение (переиспользуемый роутер).

Показывает минимальный ``APIRouter``, который можно подключить к любому своему
приложению: ``app.include_router(decision_router)``. По умолчанию — офлайн-демо
через ``TestClient``.

Запуск:

    .venv/bin/python examples/integrations/webhook_fastapi.py
"""
import sys
from pathlib import Path

from fastapi import APIRouter, FastAPI
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples._common import build_service, decision_to_dict, ticket_specs

_service = build_service()

decision_router = APIRouter(prefix="/decisions", tags=["decisions"])


class AnalyzeRequest(BaseModel):
    state: str = Field(..., min_length=1)


@decision_router.post("/analyze")
def analyze(req: AnalyzeRequest) -> dict:
    """Состояние -> типизированные решения (in-process, без внешнего сервиса)."""
    decisions = _service.decide(req.state, ticket_specs())
    return {
        "decisions": {
            name: decision_to_dict(decision) for name, decision in decisions.items()
        }
    }


def create_app() -> FastAPI:
    app = FastAPI(title="My product + decisions")
    app.include_router(decision_router)
    return app


app = create_app()


def main() -> None:
    from fastapi.testclient import TestClient

    client = TestClient(app)
    resp = client.post(
        "/decisions/analyze",
        json={"state": "Уже третьи сутки не приходит выплата Stripe, теряем выручку!"},
    )
    decisions = resp.json()["decisions"]
    print(f"[демо] HTTP {resp.status_code}, решений: {len(decisions)}")
    print(f"  команда: {decisions['команда']['choice']}")
    print(f"  срочность: {decisions['срочность']['noul']:.3f}")


if __name__ == "__main__":
    main()
