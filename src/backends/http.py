"""Бэкенд внешнего сервиса, совместимого с протоколом Jev/SystemOne.

Годится для Cloudflare Clef / Clef-Flash (Workers AI, SGLang, vLLM), Perplexity
Decider и любого другого сервиса на ``POST /v1/systemone``. Модальности
(text/image/video) объявляются вызывающей стороной и пробрасываются как есть.
"""
from __future__ import annotations

from typing import Any, Self

import httpx

from src.backends.base import (
    IMAGE,
    TEXT,
    VIDEO,
    BackendResult,
    Capabilities,
    decide_many_states,
)
from src.backends.protocol import answer_to_result, spec_to_request
from src.sampler.questions import QuestionSpec


class HttpSystemOneBackend:
    def __init__(
        self,
        base_url: str,
        model: str = "remote",
        modalities: tuple[str, ...] | list[str] = (TEXT,),
        timeout: float = 60.0,
        transport: httpx.BaseTransport | None = None,
        client: httpx.Client | None = None,
    ):
        self.name = model
        self.capabilities = Capabilities(modalities=frozenset(modalities))
        self._client = client or httpx.Client(
            base_url=base_url, timeout=timeout, transport=transport
        )

    def decide(
        self,
        state: str,
        questions: dict[str, QuestionSpec],
        *,
        images: list[Any] | None = None,
        videos: list[Any] | None = None,
        media_kwargs: dict[str, Any] | None = None,
    ) -> BackendResult:
        payload: dict[str, Any] = {
            "model": self.name,
            "state": state,
            "questions": {
                name: spec_to_request(spec) for name, spec in questions.items()
            },
        }
        if images is not None:
            payload["images"] = images
        if videos is not None:
            payload["videos"] = videos
        if media_kwargs is not None:
            payload["media_kwargs"] = media_kwargs
        resp = self._client.post("/v1/systemone", json=payload)
        resp.raise_for_status()
        body = resp.json()
        decisions = {
            name: answer_to_result(answer)
            for name, answer in body.get("answers", {}).items()
        }
        return BackendResult(decisions=decisions, usage=dict(body.get("usage", {})))

    def decide_many(
        self,
        states: list[str],
        questions: dict[str, QuestionSpec],
        *,
        images: list[Any] | None = None,
        videos: list[Any] | None = None,
        media_kwargs: dict[str, Any] | None = None,
    ) -> list[BackendResult]:
        """Внешний сервис вызывается по одному состоянию (у протокола нет батча)."""
        return decide_many_states(
            self,
            states,
            questions,
            images=images,
            videos=videos,
            media_kwargs=media_kwargs,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def parse_modalities(value: str) -> tuple[str, ...]:
    """Разбирает строку вида "text,image,video" в кортеж модальностей."""
    parts = tuple(part.strip().lower() for part in value.split(",") if part.strip())
    allowed = {TEXT, IMAGE, VIDEO}
    unknown = set(parts) - allowed
    if unknown:
        raise ValueError(f"unknown modalities: {sorted(unknown)}")
    return parts or (TEXT,)
