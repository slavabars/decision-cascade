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
from pathlib import Path
from typing import Any

from src.backends.cascade import CascadeBackend

DEFAULT_ESCALATION_COST = 0.05
DEFAULT_ERROR_COST = 1.0


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
    """

    def __init__(
        self,
        primary,
        fallback,
        remote_accuracy: float | dict[str, float],
        *,
        escalation_cost: float = DEFAULT_ESCALATION_COST,
        error_cost: float = DEFAULT_ERROR_COST,
    ):
        if escalation_cost < 0.0:
            raise ValueError(
                f"escalation_cost must be >= 0, got {escalation_cost}"
            )
        if error_cost < 0.0:
            raise ValueError(f"error_cost must be >= 0, got {error_cost}")
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
        self.name = f"benefit-cascade[{primary.name}->{fallback.name}]"

    @property
    def escalation_reason(self) -> str | None:
        return "benefit"

    def remote_accuracy_for(self, question: str) -> float:
        return self.remote_accuracy.get(question, self.default_remote_accuracy)

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
        return [
            name for name, decision in decisions.items() if self._escalate(name, decision)
        ]
