"""Mock SystemOne-сервиса на httpx.MockTransport для CI и воспроизводимых прогонов.

Ответ детерминированно зависит от (state, вопрос, seed), поэтому результаты
повторяемы без сети. Это *методологический* конкурент: он позволяет прогнать и
проверить весь конвейер (латентность, качество, расход, эскалацию), но не
претендует на качество реальных Clef/Laya/Perplexity.
"""
from __future__ import annotations

import hashlib
import json
import time
from typing import Any

import httpx


def _unit(*parts: Any) -> float:
    blob = "|".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.blake2b(blob, digest_size=8).digest()
    return int.from_bytes(digest, "big") / float(1 << 64)


def _answer(question: dict[str, Any], state: str, seed: int) -> dict[str, Any]:
    kind = question.get("type")
    instructions = question.get("instructions", "")
    if kind == "noul":
        probability = _unit(seed, state, instructions)
        return {
            "type": "noul",
            "probability": probability,
            "confidence": abs(2 * probability - 1),
        }
    keys = question.get("options") or question.get("levels") or []
    weights = [_unit(seed, state, instructions, i) for i in range(len(keys))]
    total = sum(weights) or 1.0
    probabilities = {key: weight / total for key, weight in zip(keys, weights)}
    best = max(probabilities, key=probabilities.get)
    if kind == "choice":
        return {
            "type": "choice",
            "choice": best,
            "confidence": probabilities[best],
            "probabilities": probabilities,
        }
    score = float(
        sum(i * probabilities[key] for i, key in enumerate(keys))
    )
    return {
        "type": "score",
        "score": score,
        "confidence": max(probabilities.values()),
        "probabilities": probabilities,
        "legend": list(keys),
    }


def mock_systemone_transport(
    seed: int = 0, latency_ms: float = 0.0
) -> httpx.MockTransport:
    """Транспорт, отвечающий на ``POST /v1/systemone`` детерминированно."""

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        state = payload.get("state", "")
        questions = payload.get("questions", {})
        answers = {
            name: _answer(question, state, seed)
            for name, question in questions.items()
        }
        if latency_ms:
            time.sleep(latency_ms / 1000.0)
        usage = {
            "mock": True,
            "tokens": max(1, len(state) // 4) + len(answers),
        }
        return httpx.Response(
            200,
            json={
                "model": payload.get("model", "mock"),
                "answers": answers,
                "usage": usage,
            },
        )

    return httpx.MockTransport(handler)
