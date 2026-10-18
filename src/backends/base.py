"""Абстракция бэкенда принятия решений.

Бэкенд — это исполнитель, который по состоянию и набору вопросов возвращает
типизированные решения. Реализации: локальные головы (``LocalHeadsBackend``)
и внешний сервис, совместимый с протоколом Jev/SystemOne
(``HttpSystemOneBackend``). Бэкенд объявляет ``Capabilities`` — поддерживаемые
модальности входа, — чтобы вышестоящий слой мог отклонить неподдерживаемый
запрос до его исполнения.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from src.sampler.questions import QuestionSpec
from src.schemas import DecisionResult

TEXT = "text"
IMAGE = "image"
VIDEO = "video"


@dataclass(frozen=True)
class Capabilities:
    """Что умеет бэкенд: модальности входа и приём размеченных примеров."""

    modalities: frozenset[str] = field(default_factory=lambda: frozenset({TEXT}))
    #: Принимает ли бэкенд обратную связь — размеченные примеры решений
    #: (``POST /v1/feedback``) для few-shot дообучения риск-модели под вопрос.
    #: ``False`` у всего, что не имеет риск-модели: примеры некуда применить.
    accepts_feedback: bool = False


@dataclass
class BackendResult:
    """Решения бэкенда плюс метаданные расхода/использования."""

    decisions: dict[str, DecisionResult]
    usage: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Backend(Protocol):
    name: str
    capabilities: Capabilities

    def decide(
        self,
        state: str,
        questions: dict[str, QuestionSpec],
        *,
        images: list[Any] | None = None,
        videos: list[Any] | None = None,
        media_kwargs: dict[str, Any] | None = None,
    ) -> BackendResult:
        ...

    def decide_many(
        self,
        states: list[str],
        questions: dict[str, QuestionSpec],
        *,
        images: list[Any] | None = None,
        videos: list[Any] | None = None,
        media_kwargs: dict[str, Any] | None = None,
    ) -> list[BackendResult]:
        ...


@runtime_checkable
class LocalEngine(Protocol):
    """Движок локальных голов (см. ``api.server.service.DecisionService``)."""

    @property
    def modalities(self) -> frozenset[str]:
        ...

    def decide(
        self,
        state: str,
        questions: dict[str, QuestionSpec],
        *,
        images: list[Any] | None = None,
        videos: list[Any] | None = None,
        media_kwargs: dict[str, Any] | None = None,
    ) -> dict[str, DecisionResult]:
        ...


def requested_modalities(
    *,
    has_text: bool = True,
    images: list[Any] | None = None,
    videos: list[Any] | None = None,
) -> frozenset[str]:
    """Определяет набор модальностей, фактически запрошенных во входе."""
    modalities: set[str] = set()
    if has_text:
        modalities.add(TEXT)
    if images:
        modalities.add(IMAGE)
    if videos:
        modalities.add(VIDEO)
    return frozenset(modalities)


def missing_modalities(
    requested: frozenset[str], supported: frozenset[str]
) -> frozenset[str]:
    return frozenset(requested - supported)


def decide_many_states(
    backend: Backend,
    states: list[str],
    questions: dict[str, QuestionSpec],
    *,
    images: list[Any] | None = None,
    videos: list[Any] | None = None,
    media_kwargs: dict[str, Any] | None = None,
) -> list[BackendResult]:
    """Дефолтный батч: по одному состоянию через ``decide`` (без общей батч-оптимизации)."""
    return [
        backend.decide(
            state, questions, images=images, videos=videos, media_kwargs=media_kwargs
        )
        for state in states
    ]
