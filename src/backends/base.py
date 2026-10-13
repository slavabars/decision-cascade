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
    """Поддерживаемые модальности входа; по умолчанию только текст."""

    modalities: frozenset[str] = field(default_factory=lambda: frozenset({TEXT}))


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


@runtime_checkable
class LocalEngine(Protocol):
    """Движок локальных голов (см. ``api.server.service.DecisionService``)."""

    def decide(
        self, state: str, questions: dict[str, QuestionSpec]
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
