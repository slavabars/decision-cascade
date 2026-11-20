"""Benefit-aware каскад: эскалация по ожидаемой выгоде, а не по порогу уверенности.

Порог на сыром ``confidence`` отвечает на вопрос «насколько модель уверена», а
не «стоит ли платить за внешнюю модель». Эти вопросы расходятся: при уверенности
0.7 локальная голова может ошибаться чаще внешней, а при 0.95 — ошибаться почти
никогда, и решение должно зависеть от *разницы* в качестве и цены вызова.

Политика сравнивает ожидаемую полезность двух действий. Обозначим ``p`` —
вероятность, что локальное решение верно (калиброванная уверенность), ``q`` —
вероятность, что внешняя модель на этом вопросе верна, ``D`` — цена ошибки
относительно верного ответа, ``C`` — цена эскалации в тех же единицах::

    полезность(оставить)   = p (1) + (1 - p) (-D)
    полезность(эскалировать) = q (1) + (1 - q) (-D) - C

Эскалация выгодна, когда ``p < q - C / (1 + D)``. Порог получается не из
константы, а из измеренной точности внешней модели и цены вызова: если внешняя
модель на вопросе слабее локальной (``q <= C / (1 + D)``), каскад не
эскалирует вовсе — вместо этого он молча платит за каждый вызов.

``q`` задаётся по вопросам (``remote_accuracy`` — словарь «вопрос → точность»)
или одним числом; измеряется на отложенном срезе того же домена
(``eval/competitors/fit_remote_accuracy.py``), а не на том же test, где считается
выигрыш.
"""
from __future__ import annotations

import json
import logging
import math
import random
import threading
from collections import deque
from pathlib import Path
from typing import Any

from src.backends.base import Capabilities
from src.backends.cascade import CascadeBackend
from src.feedback_store import KIND_BENEFIT, FeedbackStore, group_records

DEFAULT_ESCALATION_COST = 0.05
DEFAULT_ERROR_COST = 1.0
#: Вес априорной ``q`` в псевдонаблюдениях: обратная связь сдвигает её плавно.
DEFAULT_FEEDBACK_SHRINKAGE = 50.0
#: Сколько последних эскалаций на вопрос помнить для сопоставления с фидбэком.
DEFAULT_LEDGER_SIZE = 4096
#: Потолок веса исследовательской эскалации в IPS-оценке ``q``: ``1 / propensity``
#: при малом ``epsilon`` даёт вес 100+ и позволяет одному наблюдению сдвинуть
#: оценку. Ограничение смещает оценку только в хвосте распределения весов,
#: зато держит дисперсию под контролем.
EXPLORE_WEIGHT_CAP = 20.0

logger = logging.getLogger(__name__)


def explore_weight(propensity: float) -> float:
    """Вес наблюдения в IPS-оценке ``q``: ``1 / propensity`` с потолком.

    Эскалация по политике детерминирована (``propensity = 1``), поэтому её вес
    равен единице — при выключенном исследовании оценка совпадает с прежней
    формулой. Исследовательская эскалация происходит с вероятностью ``epsilon``,
    и её вес ``1 / epsilon`` возвращает наблюдению «представительство» всех
    похожих решений, которые политика оставила локально.
    """
    if not math.isfinite(propensity) or propensity <= 0.0:
        raise ValueError(f"propensity must be in (0, 1], got {propensity}")
    return min(1.0 / propensity, EXPLORE_WEIGHT_CAP)


def shrink_accuracy(
    prior: float, weight: float, weight_sum: float, weighted_correct: float
) -> float:
    """Байесовский сдвиг ``q`` априорной оценкой ``prior`` веса ``weight``.

    Одна формула и для счётчиков, и для IPS-весов: при ``weight_sum = total`` и
    ``weighted_correct = correct`` это прежнее
    ``(prior * W + correct) / (W + total)``.
    """
    total = weight + weight_sum
    if total <= 0.0:
        return prior
    return (prior * weight + weighted_correct) / total



def decision_fingerprint(decision: Any) -> tuple[str, Any, float]:
    """Отпечаток решения: по нему фидбэк сопоставляется с ответом fallback."""
    kind = decision.type
    if kind == "noul":
        value: Any = round(float(decision.noul), 6)
    elif kind == "choice":
        value = decision.choice
    else:
        value = round(float(decision.score), 6)
    return (kind, value, round(float(decision.confidence), 6))


def load_remote_accuracy(
    value: str, *, conservative: bool = False
) -> float | dict[str, float]:
    """Разбирает ``q`` из числа или JSON, снятого ``fit_remote_accuracy``.

    Принимает: число (``"0.86"``), путь к файлу с ``{"accuracy": ...,
    "per_question": {...}}`` или сам JSON-объект/число.

    ``conservative=True`` берёт нижнюю границу Уилсона (поле ``*_lower``), если
    она есть: на срезе в 100–200 примеров точечная точность внешней модели
    гуляет на ±0.1, и завышенная ``q`` гонит каскад в эскалации, которые
    ухудшают ответ.
    """
    text = str(value).strip()
    try:
        parsed: Any = float(text)
    except ValueError:
        path = Path(text)
        raw = path.read_text(encoding="utf-8") if path.exists() else text
        parsed = json.loads(raw)
    if isinstance(parsed, (int, float)):
        accuracy = float(parsed)
        _check_accuracy(accuracy, "remote_accuracy")
        return accuracy
    if not isinstance(parsed, dict):
        raise TypeError(f"cannot parse remote accuracy from {text!r}")
    per_question_key = "per_question_lower" if conservative else "per_question"
    accuracy_key = "accuracy_lower" if conservative else "accuracy"
    per_question_key = per_question_key if per_question_key in parsed else "per_question"
    accuracy_key = accuracy_key if accuracy_key in parsed else "accuracy"
    if per_question_key in parsed:
        per_question = {
            str(name): float(value)
            for name, value in parsed[per_question_key].items()
        }
        for name, accuracy in per_question.items():
            _check_accuracy(accuracy, f"remote_accuracy[{name}]")
        return per_question
    if accuracy_key not in parsed:
        raise ValueError("remote accuracy json needs 'accuracy' or 'per_question'")
    accuracy = float(parsed[accuracy_key])
    _check_accuracy(accuracy, "remote_accuracy")
    return accuracy


def _check_accuracy(value: float, name: str) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0.0, 1.0], got {value}")


class BenefitAwareCascadeBackend(CascadeBackend):
    """Каскад, эскалирующий только когда ожидаемая выгода превышает цену вызова.

    Наследует механику эскалации (в том числе батч и неподдерживаемые
    модальности), меняя только правило «эскалировать ли этот вопрос».

    Обратная связь уточняет ``q`` байесовским сдвигом; счётчики наблюдений при
    заданном ``feedback_path`` переживают рестарт (см. ``src.feedback_store``).
    """

    def __init__(
        self,
        primary,
        fallback,
        remote_accuracy: float | dict[str, float],
        *,
        escalation_cost: float = DEFAULT_ESCALATION_COST,
        error_cost: float = DEFAULT_ERROR_COST,
        feedback_shrinkage: float = DEFAULT_FEEDBACK_SHRINKAGE,
        ledger_size: int = DEFAULT_LEDGER_SIZE,
        feedback_path: str | Path | None = None,
        explore: float = 0.0,
        explore_seed: int | None = None,
    ):
        if escalation_cost < 0.0:
            raise ValueError(
                f"escalation_cost must be >= 0, got {escalation_cost}"
            )
        if error_cost < 0.0:
            raise ValueError(f"error_cost must be >= 0, got {error_cost}")
        if feedback_shrinkage <= 0.0:
            raise ValueError(
                f"feedback_shrinkage must be > 0, got {feedback_shrinkage}"
            )
        if ledger_size <= 0:
            raise ValueError(f"ledger_size must be > 0, got {ledger_size}")
        if not 0.0 <= explore <= 1.0:
            raise ValueError(f"explore must be in [0.0, 1.0], got {explore}")
        if isinstance(remote_accuracy, dict):
            if not remote_accuracy:
                raise ValueError("remote_accuracy dict must not be empty")
            for name, value in remote_accuracy.items():
                _check_accuracy(float(value), f"remote_accuracy[{name}]")
            self.remote_accuracy = {
                name: float(value) for name, value in remote_accuracy.items()
            }
            self.default_remote_accuracy = sum(
                self.remote_accuracy.values()
            ) / len(self.remote_accuracy)
        else:
            _check_accuracy(float(remote_accuracy), "remote_accuracy")
            self.remote_accuracy = {}
            self.default_remote_accuracy = float(remote_accuracy)
        if self.default_remote_accuracy <= 0.0:
            raise ValueError(
                "remote_accuracy must be > 0: платить за модель, которая "
                "не права никогда, бессмысленно"
            )
        super().__init__(primary, fallback, threshold=0.0)
        self.escalation_cost = float(escalation_cost)
        self.error_cost = float(error_cost)
        self.feedback_shrinkage = float(feedback_shrinkage)
        self.ledger_size = int(ledger_size)
        # Исследовательские эскалации: доля решений, которые политика оставила
        # бы локально, но мы всё равно отправляем наружу ради честной оценки q.
        self.explore = float(explore)
        self._rng = random.Random(explore_seed)
        self._pending = threading.local()
        self.explore_calls = 0
        self.name = f"benefit-cascade[{primary.name}->{fallback.name}]"
        # Обратная связь по ответам внешней модели: отпечатки выданных наружу
        # решений (чтобы связать фидбэк с fallback) и счётчики «верно/всего».
        self._remote_log: dict[
            str, deque[tuple[tuple[str, Any, float], float]]
        ] = {}
        self._observed: dict[str, list[int]] = {}
        # IPS-суммы: [сумма весов, сумма весов * верно]. Счётчики выше остаются
        # человекочитаемыми (контракт /info), оценка q идёт по весам.
        self._weighted: dict[str, list[float]] = {}
        self._lock = threading.RLock()
        self.capabilities = Capabilities(
            modalities=self.capabilities.modalities, accepts_feedback=True
        )
        # Журнал фидбэка: счётчики «верно/всего» переживают рестарт, иначе
        # политика снова опирается на срез, снятый при развёртывании.
        self.feedback_path = Path(feedback_path) if feedback_path else None
        self._feedback = (
            FeedbackStore(self.feedback_path) if self.feedback_path else None
        )
        self.restored_examples = 0
        if self._feedback is not None:
            self._restore_observations()

    def _restore_observations(self) -> None:
        """Поднять счётчики ответов внешней модели с диска.

        Журнал отпечатков (``_remote_log``) не сохраняется: он сопоставляет
        фидбэк с решениями, выданными в текущей жизни процесса. Счётчики и
        IPS-веса — это накопленная статистика, ей теряться при рестарте незачем.
        """
        for record in group_records(self._feedback.load(), KIND_BENEFIT):
            try:
                question = str(record["question"])
                correct = bool(record["correct"])
                propensity = float(record.get("propensity", 1.0))
                weight = explore_weight(propensity)
            except (KeyError, TypeError, ValueError) as exc:
                # содержимое записи не логируем: это тело фидбэка
                logger.warning(
                    "feedback: пропущено наблюдение benefit-каскада (%s)",
                    type(exc).__name__,
                )
                continue
            stats = self._observed.setdefault(question, [0, 0])
            stats[0] += 1 if correct else 0
            stats[1] += 1
            weighted = self._weighted.setdefault(question, [0.0, 0.0])
            weighted[0] += weight
            weighted[1] += weight if correct else 0.0
            self.restored_examples += 1

    @property
    def escalation_reason(self) -> str | None:
        return "benefit"

    @property
    def policy(self) -> str:
        """Имя политики для ``/info``: benefit-каскад нельзя спутать с risk."""
        return "benefit"

    def benefit_info(self) -> dict[str, Any]:
        """Параметры политики для ``GET /info -> engine``.

        Ключи — часть контракта 1.x: их можно добавлять, но не переименовывать
        и не убирать. ``remote_accuracy`` отдаётся в той форме, в которой задан
        (число или словарь по вопросам), а ``remote_accuracy_default`` — всегда
        число: это ``q`` для вопроса, которого нет в словаре.
        """
        thresholds = {
            name: self.threshold_for(name) for name in sorted(self.remote_accuracy)
        }
        with self._lock:
            feedback = {
                name: stats[1] for name, stats in sorted(self._observed.items())
            }
            explore_calls = self.explore_calls
        return {
            "policy": self.policy,
            "remote_accuracy": (
                dict(self.remote_accuracy)
                if self.remote_accuracy
                else self.default_remote_accuracy
            ),
            "remote_accuracy_default": self.default_remote_accuracy,
            "escalation_cost": self.escalation_cost,
            "error_cost": self.error_cost,
            "feedback_shrinkage": self.feedback_shrinkage,
            "benefit_threshold": self.threshold_for(""),
            "benefit_thresholds": thresholds,
            "benefit_feedback": feedback,
            "explore_rate": self.explore,
            "explore_calls": explore_calls,
        }

    def remote_accuracy_for(self, question: str) -> float:
        """``q`` для вопроса: априорная точность, сдвинутая обратной связью.

        Сдвиг байесовский: априорная ``q`` весит ``feedback_shrinkage``
        псевдонаблюдений, поэтому пара случайных ответов политику не меняет, а
        поток фидбэка — меняет. Так политика перестаёт опираться на срез,
        снятый один раз при развёртывании.

        Наблюдения взвешены по вероятности эскалации (IPS): эскалация по
        политике идёт с вероятностью 1, исследовательская — с вероятностью
        ``explore``. Без исследования вес каждого наблюдения равен 1, и оценка
        совпадает с прежней.
        """
        prior = self.remote_accuracy.get(question, self.default_remote_accuracy)
        with self._lock:
            weight_sum, weighted_correct = self._weighted.get(question, [0.0, 0.0])
        return shrink_accuracy(
            prior, self.feedback_shrinkage, weight_sum, weighted_correct
        )

    @property
    def example_counts(self) -> dict[str, int]:
        """Сколько размеченных ответов внешней модели накоплено по вопросам."""
        with self._lock:
            return {name: stats[1] for name, stats in self._observed.items()}

    def add_examples(
        self, question: str, decisions: list[Any], correct: list[Any]
    ) -> None:
        """Учесть корректность ответов, которые отдала внешняя модель.

        Фидбэк относится к решению, которое сервис вернул клиенту. Внешним оно
        было, только если отпечаток решения есть в журнале эскалаций: ответы
        локальных голов ``q`` не обновляют (они не про качество внешней модели).
        Каждое выданное решение учитывается один раз — отпечаток из журнала
        удаляется.
        """
        decisions = list(decisions)
        correct = list(correct)
        if len(decisions) != len(correct):
            raise ValueError("decisions and correct must have the same length")
        with self._lock:
            ledger = self._remote_log.get(question)
            if not ledger:
                return
            # Сопоставление идёт по копии: журнал меняется только после того,
            # как события легли на диск. Иначе ошибка записи «съела» бы
            # отпечатки, и повторный фидбэк уже не с чем было бы связать.
            remaining = list(ledger)
            accepted: list[tuple[dict[str, Any], float]] = []
            for decision, flag in zip(decisions, correct):
                fingerprint = decision_fingerprint(decision)
                index = next(
                    (
                        position
                        for position, (known, _) in enumerate(remaining)
                        if known == fingerprint
                    ),
                    None,
                )
                if index is None:
                    continue
                _, propensity = remaining.pop(index)
                accepted.append(
                    (
                        {
                            "kind": KIND_BENEFIT,
                            "question": question,
                            "correct": 1 if flag else 0,
                            "propensity": propensity,
                        },
                        propensity,
                    )
                )
            if not accepted:
                return
            if self._feedback is not None:
                self._feedback.append([record for record, _ in accepted])
            ledger.clear()
            ledger.extend(remaining)
            stats = self._observed.setdefault(question, [0, 0])
            weighted = self._weighted.setdefault(question, [0.0, 0.0])
            for record, propensity in accepted:
                correct_flag = int(record["correct"])
                weight = explore_weight(propensity)
                stats[0] += correct_flag
                stats[1] += 1
                weighted[0] += weight
                weighted[1] += weight if correct_flag else 0.0

    def _on_escalated(self, decisions: dict) -> None:
        propensities = self._pending_propensities()
        try:
            with self._lock:
                for name, decision in decisions.items():
                    propensity = float(propensities.pop(name, 1.0))
                    ledger = self._remote_log.get(name)
                    if ledger is None:
                        ledger = deque(maxlen=self.ledger_size)
                        self._remote_log[name] = ledger
                    ledger.append((decision_fingerprint(decision), propensity))
                    if propensity < 1.0:
                        self.explore_calls += 1
        finally:
            propensities.clear()

    def decide(self, *args: Any, **kwargs: Any):
        """Сбросить вероятности прошлого запроса и выполнить обычное решение."""
        self._pending_propensities().clear()
        return super().decide(*args, **kwargs)

    def decide_many(self, *args: Any, **kwargs: Any):
        """То же для батча: вероятности относятся только к текущему вызову."""
        self._pending_propensities().clear()
        return super().decide_many(*args, **kwargs)

    def _pending_propensities(self) -> dict[str, float]:
        """Вероятности эскалации решений текущего запроса (``threading.local``).

        ``_uncertain_names`` и ``_on_escalated`` вызываются в одном потоке
        одного запроса, но между ними бэкенд делает внешний вызов; локальный
        словарь не даёт параллельным запросам перепутать вероятности.
        """
        mapping = getattr(self._pending, "propensities", None)
        if mapping is None:
            mapping = {}
            self._pending.propensities = mapping
        return mapping

    def threshold_for(self, question: str) -> float:
        """``p``, ниже которого эскалация выгодна: ``q - C / (1 + D)``."""
        return self.remote_accuracy_for(question) - self.escalation_cost / (
            1.0 + self.error_cost
        )

    def expected_gain(self, question: str, confidence: float) -> float:
        """Выгода эскалации в единицах «верный ответ = 1».

        Положительная — платить за внешнюю модель стоит, отрицательная — нет.
        """
        q = self.remote_accuracy_for(question)
        return (q - confidence) - self.escalation_cost / (1.0 + self.error_cost)

    def _escalate(self, question: str, decision: Any) -> bool:
        return self.expected_gain(question, float(decision.confidence)) > 0.0

    def _uncertain(self, decision) -> bool:
        return self.expected_gain("", float(decision.confidence)) > 0.0

    def _uncertain_names(self, decisions: dict) -> list[str]:
        """Вопросы на эскалацию: политика плюс исследовательская доля.

        Решение, которое политика оставила бы локально, уходит наружу с
        вероятностью ``explore``. Это нужно не для качества ответа, а для
        оценки ``q``: фидбэк приходит только по эскалациям, поэтому без
        исследования мы видим качество внешней модели лишь на трудных случаях.
        Вероятность эскалации запоминается для IPS-весов в
        ``_on_escalated``/``add_examples``; при ``explore = 0`` поведение
        прежнее, и ``q`` считается по одним счётчикам.
        """
        propensities = self._pending_propensities()
        propensities.clear()
        names: list[str] = []
        with self._lock:
            for name, decision in decisions.items():
                if self._escalate(name, decision):
                    names.append(name)
                    propensities[name] = 1.0
                elif self.explore > 0.0 and self._rng.random() < self.explore:
                    names.append(name)
                    propensities[name] = self.explore
        return names
