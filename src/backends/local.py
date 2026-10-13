"""Локальный бэкенд: решения считаются встроенными головами на CPU/GPU."""
from __future__ import annotations

from typing import Any

from src.backends.base import BackendResult, Capabilities, LocalEngine
from src.sampler.questions import QuestionSpec


class LocalHeadsBackend:
    """Оборачивает локальный движок голов; поддерживает только текст."""

    def __init__(self, engine: LocalEngine, name: str = "local"):
        self.engine = engine
        self.name = name
        self.capabilities = Capabilities()

    def decide(
        self,
        state: str,
        questions: dict[str, QuestionSpec],
        *,
        images: list[Any] | None = None,
        videos: list[Any] | None = None,
        media_kwargs: dict[str, Any] | None = None,
    ) -> BackendResult:
        del images, videos, media_kwargs  # текстовая модальность; медиа отклоняются выше
        decisions = self.engine.decide(state, questions)
        return BackendResult(
            decisions=decisions,
            usage={"questions": len(questions)},
        )
