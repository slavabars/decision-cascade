"""Локальный бэкенд: решения считаются встроенными головами на CPU/GPU."""
from __future__ import annotations

from typing import Any

from src.backends.base import TEXT, BackendResult, Capabilities, LocalEngine
from src.sampler.questions import QuestionSpec


class LocalHeadsBackend:
    """Оборачивает локальный движок голов и объявляет его модальности."""

    def __init__(self, engine: LocalEngine, name: str = "local"):
        self.engine = engine
        self.name = name
        supported = getattr(engine, "modalities", frozenset({TEXT}))
        self.capabilities = Capabilities(modalities=frozenset(supported))

    def decide(
        self,
        state: str,
        questions: dict[str, QuestionSpec],
        *,
        images: list[Any] | None = None,
        videos: list[Any] | None = None,
        media_kwargs: dict[str, Any] | None = None,
    ) -> BackendResult:
        decisions = self.engine.decide(
            state,
            questions,
            images=images,
            videos=videos,
            media_kwargs=media_kwargs,
        )
        return BackendResult(
            decisions=decisions,
            usage={"questions": len(questions)},
        )
