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
from src.errors import UntrainedDecisionError
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

    def untrained_falls_back(self) -> bool:
        """Эскалировать ли во внешнюю модель, если primary не обучен.

        Имеет смысл только при политике ``fallback``: тогда сервис сигналит
        ``UntrainedDecisionError``, и каскад обязан обслужить запрос удалённо.
        При ``refuse`` исключение пробрасывается наружу (API вернёт 409).
        """
        engine = getattr(self.primary, "engine", None)
        return getattr(engine, "untrained_policy", None) == "fallback"

    def _uncertain(self, decision) -> bool:
        """Эскалировать ли отдельное решение. В базовом каскаде — порог уверенности."""
        return decision.confidence < self.threshold

    def _uncertain_names(self, decisions: dict) -> list[str]:
        return [name for name, d in decisions.items() if self._uncertain(d)]

    @property
    def escalation_reason(self) -> str | None:
        """Причина эскалации для ``usage`` (у базового каскада — не фиксируется)."""
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

        try:
            primary_result = self.primary.decide(
                state, questions, images=images, videos=videos, media_kwargs=media_kwargs
            )
        except UntrainedDecisionError:
            if not self.untrained_falls_back():
                raise
            result = self.fallback.decide(
                state,
                questions,
                images=images,
                videos=videos,
                media_kwargs=media_kwargs,
            )
            usage = self._merge_usage(
                {}, result.usage, len(questions), reason="untrained"
            )
            return BackendResult(decisions=dict(result.decisions), usage=usage)
        uncertain = self._uncertain_names(primary_result.decisions)
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
            primary_result.usage,
            fallback_result.usage,
            len(uncertain),
            reason=self.escalation_reason,
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

        try:
            primary_results = _decide_many(
                self.primary,
                states,
                questions,
                images=images,
                videos=videos,
                media_kwargs=media_kwargs,
            )
        except UntrainedDecisionError:
            if not self.untrained_falls_back():
                raise
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
                        {}, result.usage, len(questions), reason="untrained"
                    ),
                )
                for result in fallback_results
            ]
        results: list[BackendResult] = []
        for state, primary_result in zip(states, primary_results):
            uncertain = self._uncertain_names(primary_result.decisions)
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
                        reason=self.escalation_reason,
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


class RiskAwareCascadeBackend(CascadeBackend):
    """Каскад, эскалирующий по оценке вероятности ошибки, а не по уверенности.

    ``ErrorRiskModel`` предсказывает ``P(ошибка)`` по признакам решения
    (уверенность, энтропия, отрыв), и решение уходит во внешнюю модель, когда
    оценка превышает ``risk_budget``. Бюджет задаётся в терминах *доли ошибок*,
    которые мы готовы принять локально, — в отличие от порога на сыром
    confidence, который не равен вероятности ошибки.

    **Hard gate.** Если у модели есть заключение гейта с ``accepted: false``
    (risk-вариант измерен и оказался не лучше baseline), каскад безусловно
    фолбэчит на порог уверенности (``fallback_threshold``): причина эскалации в
    ``usage`` становится ``risk_gate`` — так видно, что фолбэк применён.
    Отсутствие заключения означает, что замера не было, и режим работает
    как прежде.
    """

    def __init__(
        self,
        primary: Backend,
        fallback: Backend,
        risk_model,
        risk_budget: float = 0.05,
        *,
        gate: dict | None = None,
        fallback_threshold: float = DEFAULT_THRESHOLD,
    ):
        if not 0.0 <= risk_budget <= 1.0:
            raise ValueError(f"risk_budget must be in [0.0, 1.0], got {risk_budget}")
        if not 0.0 <= fallback_threshold <= 1.0:
            raise ValueError(
                f"fallback_threshold must be in [0.0, 1.0], got {fallback_threshold}"
            )
        super().__init__(primary, fallback, threshold=0.0)
        self.risk_model = risk_model
        self.risk_budget = risk_budget
        gate = gate if gate is not None else getattr(risk_model, "gate", None)
        self.gate = gate
        self.gate_rejected = bool(gate) and not bool(gate.get("accepted", True))
        self.fallback_threshold = fallback_threshold
        self.name = f"risk-cascade[{primary.name}->{fallback.name}]"

    @property
    def escalation_reason(self) -> str | None:
        if self.gate_rejected:
            return "risk_gate"
        return "risk"

    def _uncertain(self, decision) -> bool:
        if self.gate_rejected:
            # решение худшего варианта отклонено: работает лучший baseline
            return decision.confidence < self.fallback_threshold
        return bool(self.risk_model.should_escalate(decision, self.risk_budget))
