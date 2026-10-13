"""Минимальный клиент протокола SystemOne (чистый httpx, без нашего SDK).

Показывает, что внешний сервис (Clef/Clef-Flash, vLLM, SGLang и т.п.) и наш
сервер вызываются одинаково. Вопросы задаются через ``criteria`` — опции и
уровни выводятся из них.

Запуск против нашего сервера:

    .venv/bin/python -m uvicorn api.server.main:app --port 8000
    .venv/bin/python examples/protocol/systemone_client.py

Запуск против внешнего сервиса:

    .venv/bin/python examples/protocol/systemone_client.py \\
        --base-url http://127.0.0.1:30000 --model clef-flash
"""
import argparse
import sys
from pathlib import Path
from typing import Any, Self

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

STATE = "Не приходит выплата Stripe третьи сутки, теряем выручку!"

QUESTIONS: dict[str, dict[str, Any]] = {
    "team": {
        "type": "choice",
        "criteria": {
            "billing": "оплата и списания",
            "integration": "API и вебхуки",
            "sales": "тарифы и лицензии",
            "support": "интерфейс и гайды",
        },
    },
    "urgent": {"type": "noul", "criteria": ["нет спешки", "срочно"]},
    "severity": {
        "type": "score",
        "criteria": ["спокоен", "недоволен", "в ярости"],
    },
}


class SystemOneClient:
    """Тонкая обёртка над ``POST /v1/systemone``."""

    def __init__(self, base_url: str = "http://localhost:8000", timeout: float = 60.0):
        self._client = httpx.Client(base_url=base_url, timeout=timeout)

    def decide(
        self,
        state: str,
        questions: dict[str, dict[str, Any]],
        *,
        model: str | None = None,
        images: list[Any] | None = None,
        videos: list[Any] | None = None,
        media_kwargs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"state": state, "questions": questions}
        if model:
            payload["model"] = model
        if images:
            payload["images"] = images
        if videos:
            payload["videos"] = videos
        if media_kwargs:
            payload["media_kwargs"] = media_kwargs
        resp = self._client.post("/v1/systemone", json=payload)
        resp.raise_for_status()
        return resp.json()

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="SystemOne protocol client")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--model", default=None, help="поле model для внешнего сервиса")
    args = parser.parse_args()

    with SystemOneClient(base_url=args.base_url) as client:
        body = client.decide(STATE, QUESTIONS, model=args.model)

    print(f"server model: {body['model']}")
    print(f"usage: {body['usage']}\n")
    for name, answer in body["answers"].items():
        kind = answer["type"]
        if kind == "noul":
            print(f"{name:<10} noul={answer['probability']:.3f}")
        elif kind == "choice":
            print(f"{name:<10} choice={answer['choice']!r}")
        else:
            print(f"{name:<10} score={answer['score']:.3f}")


if __name__ == "__main__":
    main()
