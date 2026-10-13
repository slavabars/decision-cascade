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
    decide_many_states,
    missing_modalities,
    requested_modalities,
)
from src.sampler.questions import QuestionSpec

DEFAULT_THRESHOLD = 0.6


def _decide_many(
    backend: Backend,
    states: list[str],
    questions: dict[str, QuestionSpec],
    *,
    images: list[Any] | None = None,
    videos: list[Any] | None = None,
    media_kwargs: dict[str, Any] | None = None,
) -> list[BackendResult]:
    """Батч через ``decide_many`` бэкенда, если он есть, иначе — по одному состоянию."""
    many = getattr(backend, "decide_many", None)
    if callable(many):
        return many(
            states, questions, images=images, videos=videos, media_kwargs=media_kwargs
        )
    return decide_many_states(
        backend,
        states,
        questions,
        images=images,
        videos=videos,
        media_kwargs=media_kwargs,
    )


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

    def decide_many(
        self,
        states: list[str],
        questions: dict[str, QuestionSpec],
        *,
        images: list[Any] | None = None,
        videos: list[Any] | None = None,
        media_kwargs: dict[str, Any] | None = None,
    ) -> list[BackendResult]:
        """Батч-версия каскада: primary идёт одним набором, эскалируются неясные."""
        if not states:
            return []
        requested = requested_modalities(images=images, videos=videos)
        if missing_modalities(requested, self.primary.capabilities.modalities):
            fallback_results = _decide_many(
                self.fallback,
                states,
                questions,
                images=images,
                videos=videos,
                media_kwargs=media_kwargs,
            )
            return [
                BackendResult(
                    decisions=dict(result.decisions),
                    usage=self._merge_usage(
                        {}, result.usage, len(questions), reason="modality"
                    ),
                )
                for result in fallback_results
            ]

        primary_results = _decide_many(
            self.primary,
            states,
            questions,
            images=images,
            videos=videos,
            media_kwargs=media_kwargs,
        )
        results: list[BackendResult] = []
        for state, primary_result in zip(states, primary_results):
            uncertain = [
                name
                for name, decision in primary_result.decisions.items()
                if decision.confidence < self.threshold
            ]
            if not uncertain:
                usage = dict(primary_result.usage)
                usage["escalated"] = 0
                results.append(
                    BackendResult(decisions=dict(primary_result.decisions), usage=usage)
                )
                continue
            subset = {name: questions[name] for name in uncertain}
            fallback_result = self.fallback.decide(
                state, subset, images=images, videos=videos, media_kwargs=media_kwargs
            )
            merged = dict(primary_result.decisions)
            merged.update(fallback_result.decisions)
            results.append(
                BackendResult(
                    decisions=merged,
                    usage=self._merge_usage(
                        primary_result.usage,
                        fallback_result.usage,
                        len(uncertain),
                    ),
                )
            )
        return results

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
