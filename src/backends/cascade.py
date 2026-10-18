"""Каскадный бэкенд: сначала локальные головы, при неуверенности — внешняя модель.

Идея: дешёвый локальный бэкенд решает всё, что умеет уверенно; вопросы с низкой
уверенностью (или неподдерживаемую модальность) эскалируются во внешний
SystemOne-сервис. Так экономится основная часть внешних вызовов.
"""
from __future__ import annotations

import threading
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
from src.risk import features_matrix, gate_verdict
from src.sampler.questions import QuestionSpec

DEFAULT_THRESHOLD = 0.6


def _gate_verdict(
    gate: dict | None, delta: float, risk_model: Any
) -> tuple[bool, list[str]]:
    """Вердикт hard gate с допуском ``delta`` (см. ``src.risk.gate_verdict``).

    Явно переданный ``gate`` имеет приоритет над сертификатом модели: тогда
    считаем его напрямую. Если взят сертификат самой модели и она умеет
    ``gate_verdict`` — делегируем ей (учитываются по-метричные ``excess``).
    """
    if gate is not None and gate is getattr(risk_model, "gate", None):
        verdict = getattr(risk_model, "gate_verdict", None)
        if callable(verdict):
            return verdict(delta)
    return gate_verdict(gate, delta)


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

    **Few-shot (Шаг 6).** Для *нового* вопроса коэффициент, снятый с других
    вопросов, переносится плохо, поэтому риск-модель на нём надо переобучать.
    ``add_examples`` копит размеченные примеры вопроса и, набрав ``few_shot_n``,
    переобучает логистическую регрессию по схеме {``reference`` + свои}
    (numpy, CPU-only — см. ``src.risk.fit_logistic``); переобученная модель
    сразу применяется к этому вопросу. Hard gate остаётся жёстким: он проверяется
    на уровне каскада, и few-shot его не обходит — если сертификат отклонён,
    работает ``confidence`` и для переобученного вопроса.

    Транспорт примеров — ``POST /v1/feedback`` (см. ``api/server/main.py``):
    клиент присылает решения, которые сервис ему отдал, с разметкой корректности.
    Пример — это ``(вопрос, решение, корректность)``: признаки риск-модели
    выводятся из одного решения (``src.risk.decision_features``), поэтому
    достаточно повторить решение. Копилка — на экземпляре каскада, доступ к ней
    закрыт блокировкой (``example_counts``/``add_examples``/``reference_rows``
    потокобезопасны).

    ``reference`` по умолчанию строится из **накопленных примеров других
    вопросов** — тот же смысл, что у ``others`` в офлайновой кривой few-shot
    (``eval.risk.features.shots_report``), то есть сервис воспроизводит
    измеренную схему {чужие строки + свои}. Явно переданный ``reference`` имеет
    приоритет над выводом из копилки.
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
        gate_delta: float = 0.0,
        few_shot_n: int | None = None,
        reference: tuple[Any, Any] | None = None,
    ):
        if not 0.0 <= risk_budget <= 1.0:
            raise ValueError(f"risk_budget must be in [0.0, 1.0], got {risk_budget}")
        if not 0.0 <= fallback_threshold <= 1.0:
            raise ValueError(
                f"fallback_threshold must be in [0.0, 1.0], got {fallback_threshold}"
            )
        if few_shot_n is not None and few_shot_n < 2:
            raise ValueError(f"few_shot_n must be >= 2 or None, got {few_shot_n}")
        super().__init__(primary, fallback, threshold=0.0)
        self.risk_model = risk_model
        self.risk_budget = risk_budget
        gate = gate if gate is not None else getattr(risk_model, "gate", None)
        self.gate = gate
        self.gate_delta = float(gate_delta)
        self.gate_accepted, self.gate_worse = _gate_verdict(
            gate, self.gate_delta, risk_model
        )
        self.gate_rejected = not self.gate_accepted
        self.fallback_threshold = fallback_threshold
        # few-shot: порог, общие строки {reference} и накопленные свои примеры
        self.few_shot_n = few_shot_n
        self.reference = reference
        self._examples: dict[str, list[tuple[Any, bool]]] = {}
        self._few_shot_models: dict[str, Any] = {}
        # Копилка примеров и переобученные модели правятся из эндпоинта фидбэка,
        # который FastAPI выполняет в threadpool, — общий доступ прикрываем.
        self._lock = threading.RLock()
        self.name = f"risk-cascade[{primary.name}->{fallback.name}]"
        # Фидбэк осмыслен, только когда few-shot включён: иначе примеры некуда
        # применить (``add_examples`` — no-op).
        self.capabilities = Capabilities(
            modalities=self.capabilities.modalities,
            accepts_feedback=self.few_shot_n is not None,
        )

    @property
    def escalation_reason(self) -> str | None:
        if self.gate_rejected:
            return "risk_gate"
        return "risk"

    @property
    def few_shot_questions(self) -> list[str]:
        """Вопросы, для которых уже переобучена few-shot модель."""
        with self._lock:
            return sorted(self._few_shot_models)

    @property
    def example_counts(self) -> dict[str, int]:
        """Сколько размеченных примеров накоплено по каждому вопросу."""
        with self._lock:
            return {
                question: len(bucket)
                for question, bucket in self._examples.items()
            }

    def add_examples(
        self, question: str, decisions: list[Any], correct: list[Any]
    ) -> Any | None:
        """Накопить размеченные примеры вопроса; при ``few_shot_n`` — переобучить.

        ``correct`` — факт корректности решения (1/True = верно), как в
        ``training.recipe.risk``. Возвращает переобученную модель, если порог
        достигнут (она сразу используется для этого вопроса), иначе ``None``.
        Блокировка держится на всё тело метода: приём примеров и чтение копилки
        идут из разных потоков (эндпоинт фидбэка против решений), поэтому
        копить и обучать нужно во взаимно исключающем порядке.
        """
        if self.few_shot_n is None:
            return None
        decisions = list(decisions)
        correct = list(correct)
        if len(decisions) != len(correct):
            raise ValueError("decisions and correct must have the same length")
        with self._lock:
            bucket = self._examples.setdefault(question, [])
            bucket.extend(zip(decisions, correct))
            if len(bucket) < self.few_shot_n:
                return None
            own_features = features_matrix([decision for decision, _ in bucket])
            own_labels = [1 if flag else 0 for _, flag in bucket]
            reference_features, reference_labels = self.reference_rows(question)
            model = self.risk_model.retrained_with(
                own_features,
                own_labels,
                reference_features=reference_features,
                reference_labels=reference_labels,
            )
            self._few_shot_models[question] = model
            return model

    def reference_rows(self, question: str) -> tuple[Any, Any] | tuple[None, None]:
        """Строки «чужих» вопросов под схему {``reference`` + свои}.

        Явно переданный ``reference`` (матрица признаков и метки отложенного
        набора, на которых обучена базовая модель) имеет приоритет. Иначе роль
        чужих строк играют накопленные примеры **других** вопросов — тот же смысл,
        что у ``others`` в офлайновой кривой few-shot
        (``eval.risk.features.shots_report``), поэтому в сервисе воспроизводится
        именно измеренная схема, а не упрощённая. Пока чужих примеров нет,
        возвращается ``(None, None)`` и обучение идёт только на своих
        (см. ``ErrorRiskModel.retrained_with``) — режим измерен отдельно
        (см. ``docs/risk-aware-cascade.md``, «Нужны ли чужие строки»): он хуже,
        пока своих примеров мало, поэтому эндпоинт фидбэка отдаёт размер
        ``reference``, чтобы режим был виден клиенту.

        Публичный: и ``add_examples`` (под блокировкой), и эндпоинт фидбэка
        читают копилку — метод защищает её сам, ``RLock`` реентерабельна.
        """
        if self.reference is not None:
            return self.reference
        with self._lock:
            others = [
                (decision, flag)
                for name, bucket in self._examples.items()
                if name != question
                for decision, flag in bucket
            ]
        if not others:
            return None, None
        return (
            features_matrix([decision for decision, _ in others]),
            [1 if flag else 0 for _, flag in others],
        )

    def _model_for(self, question: str) -> Any:
        with self._lock:
            return self._few_shot_models.get(question, self.risk_model)

    def _escalate(self, model: Any, decision: Any) -> bool:
        if self.gate_rejected:
            # решение худшего варианта отклонено: работает лучший baseline
            return decision.confidence < self.fallback_threshold
        return bool(model.should_escalate(decision, self.risk_budget))

    def _uncertain(self, decision) -> bool:
        return self._escalate(self.risk_model, decision)

    def _uncertain_names(self, decisions: dict) -> list[str]:
        return [
            name
            for name, decision in decisions.items()
            if self._escalate(self._model_for(name), decision)
        ]
