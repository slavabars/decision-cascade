"""Локальный бэкенд: решения считаются встроенными головами на CPU/GPU."""
from __future__ import annotations

from typing import Any

from src.backends.base import (
    TEXT,
    BackendResult,
    Capabilities,
    LocalEngine,
    backend_readiness,
    decide_many_states,
)
from src.sampler.questions import QuestionSpec


class LocalHeadsBackend:
    """Оборачивает локальный движок голов и объявляет его модальности."""

    def __init__(self, engine: LocalEngine, name: str = "local"):
        self.engine = engine
        self.name = name
        supported = getattr(engine, "modalities", frozenset({TEXT}))
        self.capabilities = Capabilities(modalities=frozenset(supported))

    def readiness(self) -> dict[str, Any]:
        """Готовность локального движка: загружен ли энкодер и есть ли головы."""
        report = backend_readiness(self.engine)
        report.setdefault("backend", self.name)
        return report

    def _trained(self, questions: dict[str, QuestionSpec]) -> bool | None:
        """Обучены ли головы под схему (``None`` — движок не сообщает)."""
        is_trained = getattr(self.engine, "is_trained", None)
        if callable(is_trained):
            return bool(is_trained(questions))
        return None

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
            usage={"questions": len(questions), "trained": self._trained(questions)},
        )

    def decide_many(
        self,
        states: list[str],
        questions: dict[str, QuestionSpec],
        *,
        images: list[Any] | None = None,
        videos: list[Any] | None = None,
        media_kwargs: dict[str, Any] | None = None,
    ) -> list[BackendResult]:
        """Батч: если движок умеет ``decide_many`` (без медиа) — один прогон энкодера."""
        engine_many = getattr(self.engine, "decide_many", None)
        trained = self._trained(questions)
        text_only = images is None and videos is None and media_kwargs is None
        if engine_many is not None and text_only:
            return [
                BackendResult(
                    decisions=decisions,
                    usage={"questions": len(questions), "trained": trained},
                )
                for decisions in engine_many(states, questions)
            ]
        return decide_many_states(
            self,
            states,
            questions,
            images=images,
            videos=videos,
            media_kwargs=media_kwargs,
        )
