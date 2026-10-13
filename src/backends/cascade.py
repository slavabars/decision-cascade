"""Каскадный бэкенд: сначала локальные головы, при неуверенности — внешняя модель.

Идея: дешёвый локальный бэкенд решает всё, что умеет уверенно; вопросы с низкой
уверенностью (или неподдерживаемую модальность) эскалируются во внешний
SystemOne-сервис. Так экономится основная часть внешних вызовов.
"""
from __future__ import annotations

from typing import Any, Self

from src.backends.base import (
    Backend,
    BackendResult,
    Capabilities,
    missing_modalities,
    requested_modalities,
)
from src.sampler.questions import QuestionSpec

DEFAULT_THRESHOLD = 0.6


class CascadeBackend:
    def __init__(
        self, primary: Backend, fallback: Backend, threshold: float = DEFAULT_THRESHOLD
    ):
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be in [0.0, 1.0], got {threshold}")
        self.primary = primary
        self.fallback = fallback
        self.threshold = threshold
        self.name = f"cascade[{primary.name}->{fallback.name}]"
        self.capabilities = Capabilities(
            modalities=primary.capabilities.modalities
            | fallback.capabilities.modalities
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
        requested = requested_modalities(images=images, videos=videos)
        if missing_modalities(requested, self.primary.capabilities.modalities):
            result = self.fallback.decide(
                state,
                questions,
                images=images,
                videos=videos,
                media_kwargs=media_kwargs,
            )
            usage = self._merge_usage({}, result.usage, len(questions), reason="modality")
            return BackendResult(decisions=dict(result.decisions), usage=usage)

        primary_result = self.primary.decide(
            state, questions, images=images, videos=videos, media_kwargs=media_kwargs
        )
        uncertain = [
            name
            for name, decision in primary_result.decisions.items()
            if decision.confidence < self.threshold
        ]
        if not uncertain:
            usage = dict(primary_result.usage)
            usage["escalated"] = 0
            return BackendResult(
                decisions=dict(primary_result.decisions), usage=usage
            )

        subset = {name: questions[name] for name in uncertain}
        fallback_result = self.fallback.decide(
            state, subset, images=images, videos=videos, media_kwargs=media_kwargs
        )
        merged = dict(primary_result.decisions)
        merged.update(fallback_result.decisions)
        usage = self._merge_usage(
            primary_result.usage, fallback_result.usage, len(uncertain)
        )
        return BackendResult(decisions=merged, usage=usage)

    @staticmethod
    def _merge_usage(
        primary_usage: dict[str, Any],
        fallback_usage: dict[str, Any],
        escalated: int,
        *,
        reason: str | None = None,
    ) -> dict[str, Any]:
        usage = dict(primary_usage)
        usage["fallback"] = dict(fallback_usage)
        usage["escalated"] = escalated
        if reason is not None:
            usage["reason"] = reason
        return usage

    def close(self) -> None:
        for backend in (self.primary, self.fallback):
            close = getattr(backend, "close", None)
            if callable(close):
                close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
