"""Клиентский каскад: primary -> fallback по уверенности.

Серверный каскад (``DECISION_BACKEND=cascade``) эскалирует внутри себя, но клиент
может делать это и сам — удобно, когда primary и fallback разнородны (наш сервер
и внешний LLM). Пример запрашивает primary; если минимальная уверенность ответов
ниже порога, повторяет запрос к fallback.

Требует запущенные серверы:

    .venv/bin/python -m uvicorn api.server.main:app --port 8000
    .venv/bin/python -m uvicorn api.server.main:app --port 8001   # «внешний»
    .venv/bin/python examples/cascade/client_fallback.py --fallback http://localhost:8001
"""
import argparse
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples._common import ticket_questions

STATE = "Не приходит выплата Stripe третьи сутки, теряем выручку!"


def decide(base_url: str, state: str) -> dict:
    with httpx.Client(base_url=base_url, timeout=60.0) as client:
        resp = client.post(
            "/v1/systemone", json={"state": state, "questions": ticket_questions()}
        )
        resp.raise_for_status()
        return resp.json()


def min_confidence(body: dict) -> float:
    answers = body.get("answers", {})
    if not answers:
        return 0.0
    return min(float(a.get("confidence", 0.0)) for a in answers.values())


def main() -> None:
    parser = argparse.ArgumentParser(description="Client-side cascade by confidence")
    parser.add_argument("--primary", default="http://localhost:8000")
    parser.add_argument("--fallback", default=None, help="URL fallback-сервиса")
    parser.add_argument("--threshold", type=float, default=0.6)
    args = parser.parse_args()

    primary = decide(args.primary, STATE)
    confidence = min_confidence(primary)
    print(f"primary {args.primary}: model={primary['model']}, "
          f"min confidence={confidence:.3f}")

    if confidence >= args.threshold or not args.fallback:
        if confidence < args.threshold:
            print(f"ниже порога {args.threshold}, но fallback не задан — ответ primary")
        print("Ответ primary принят.")
        return

    print(f"ниже порога {args.threshold} -> эскалация в {args.fallback}")
    fallback = decide(args.fallback, STATE)
    print(f"fallback {args.fallback}: model={fallback['model']}, "
          f"min confidence={min_confidence(fallback):.3f}")


if __name__ == "__main__":
    main()
