"""Смещение оценки точности внешней модели при фидбэке только по эскалациям.

Политика benefit эскалирует вопрос, когда локальная уверенность ниже порога
``q - C / (1 + D)``. Фидбэк приходит по ответам внешней модели, то есть почти
всегда по эскалированным решениям — а это трудные случаи. Если трудность
коррелирует с качеством внешней модели, оценка ``q`` смещается, порог уезжает,
и цикл замыкается: чем ниже оценка, тем меньше эскалаций, тем труднее выборка.

Симуляция изолирует именно оценщик, а не качество моделей. Строки генерируются
из двух страт: «лёгкие» (высокая локальная уверенность, точность внешней модели
``q_easy``) и «трудные» (низкая уверенность, ``q_hard``). Политика видит только
уверенность, а правильность внешней модели зависит от страты. Сравниваются:

- ``naive`` — каждое наблюдение весит 1 (текущее поведение без исследования);
- ``ips`` — вес ``1 / propensity`` с потолком (эскалация по политике — 1,
  исследовательская — ``1 / epsilon``).

Исследовательская эскалация отправляет наружу долю решений, которые политика
оставила бы локально: без неё у локальных решений нет наблюдений вообще, и
никакое взвешивание не помогает. ``naive`` при ``epsilon > 0`` показывает, что
одной случайной подвыборки мало: без весов оценка остаётся смещённой.

Формула сдвига и вес берутся из ``src.backends.benefit`` — симуляция считает
ровно то, что считает сервис. Полезность считается в единицах «верный ответ = 1»
при допущении, что локальная уверенность калибрована: оставить локально даёт
``p``, эскалация — ``q`` за вычетом цены вызова. ``regret`` — потеря относительно
оракула, который знает качество внешней модели на каждой строке.
"""
from __future__ import annotations

import random
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from src.backends.benefit import explore_weight, shrink_accuracy

#: Начальная ``q`` до обратной связи: срез, снятый при развёртывании.
DEFAULT_PRIOR = 0.75
#: Вес априорной оценки в псевдонаблюдениях (как ``feedback_shrinkage``).
DEFAULT_SHRINKAGE = 50.0
DEFAULT_COST = 0.05
DEFAULT_ERROR_COST = 1.0


@dataclass(frozen=True)
class Scenario:
    """Две страты решений: локальная уверенность и качество внешней модели."""

    name: str
    q_easy: float
    q_hard: float
    share_hard: float
    description: str

    @property
    def q_true(self) -> float:
        """Точность внешней модели на всей популяции решений."""
        return self.share_hard * self.q_hard + (1.0 - self.share_hard) * self.q_easy


#: ``correlated`` — трудные для локальной модели случаи трудны и для внешней
#: (оценка ``q`` занижается). ``anti`` — внешняя модель сильнее именно на
#: трудных (оценка завышается). Направление смещения — свойство домена, поэтому
#: проверяются оба знака.
SCENARIOS: dict[str, Scenario] = {
    "correlated": Scenario(
        name="correlated",
        q_easy=0.95,
        q_hard=0.60,
        share_hard=0.35,
        description="трудные для локальной модели случаи трудны и для внешней",
    ),
    "anti": Scenario(
        name="anti",
        q_easy=0.70,
        q_hard=0.95,
        share_hard=0.35,
        description="внешняя модель сильнее именно на трудных случаях",
    ),
}


@dataclass(frozen=True)
class Row:
    """Одно решение: уверенность, исход и ожидаемое качество внешней модели."""

    confidence: float
    correct: bool
    external_q: float
    hard: bool
    draw: float


def make_rows(
    scenario: Scenario, rng: random.Random, rows: int
) -> list[Row]:
    """Сгенерировать строки одной волны: страта -> уверенность -> исход."""
    result: list[Row] = []
    for _ in range(rows):
        hard = rng.random() < scenario.share_hard
        if hard:
            confidence = rng.uniform(0.45, 0.64)
            external_q = scenario.q_hard
            correct = rng.random() < scenario.q_hard
        else:
            confidence = rng.uniform(0.65, 0.95)
            external_q = scenario.q_easy
            correct = rng.random() < scenario.q_easy
        result.append(
            Row(
                confidence=confidence,
                correct=correct,
                external_q=external_q,
                hard=hard,
                draw=rng.random(),
            )
        )
    return result


def run_policy(
    rounds: list[list[Row]],
    *,
    estimator: str,
    explore: float,
    q_true: float,
    prior: float = DEFAULT_PRIOR,
    shrinkage: float = DEFAULT_SHRINKAGE,
    cost: float = DEFAULT_COST,
    error_cost: float = DEFAULT_ERROR_COST,
) -> dict[str, Any]:
    """Прогнать политику по волнам строк и вернуть итоговую оценку ``q``."""
    if estimator not in {"naive", "ips"}:
        raise ValueError(f"estimator must be 'naive' or 'ips', got {estimator!r}")
    if not 0.0 <= explore <= 1.0:
        raise ValueError(f"explore must be in [0.0, 1.0], got {explore}")
    weight_sum = 0.0
    weighted_correct = 0.0
    q_hat = prior
    calls = 0
    policy_escalations = 0
    total = 0
    utility = 0.0
    oracle_utility = 0.0
    trace: list[float] = []
    for rows in rounds:
        threshold = q_hat - cost / (1.0 + error_cost)
        for row in rows:
            policy = row.confidence < threshold
            explored = (not policy) and row.draw < explore
            if policy:
                policy_escalations += 1
            if policy or explored:
                calls += 1
                utility += row.external_q - cost
            else:
                utility += row.confidence
            # оракул знает качество внешней модели на строке, а не оценку
            oracle_escalate = (
                row.confidence < row.external_q - cost / (1.0 + error_cost)
            )
            oracle_utility += (
                row.external_q - cost if oracle_escalate else row.confidence
            )
            if not (policy or explored):
                continue
            if estimator == "naive":
                weight = 1.0
            else:
                weight = explore_weight(1.0 if policy else explore)
            weight_sum += weight
            if row.correct:
                weighted_correct += weight
        total += len(rows)
        q_hat = shrink_accuracy(prior, shrinkage, weight_sum, weighted_correct)
        trace.append(q_hat)
    return {
        "q_hat": q_hat,
        "bias": q_hat - q_true,
        "calls": calls,
        "policy_escalations": policy_escalations,
        "total": total,
        "calls_rate": calls / total if total else 0.0,
        "policy_rate": policy_escalations / total if total else 0.0,
        "utility_rate": utility / total if total else 0.0,
        "oracle_rate": oracle_utility / total if total else 0.0,
        "regret_rate": (
            (oracle_utility - utility) / total if total else 0.0
        ),
        "weight_sum": weight_sum,
        "trace": trace,
    }


def simulate(
    scenario: Scenario,
    *,
    estimator: str,
    explore: float,
    seeds: Iterable[int] = range(5),
    rounds: int = 30,
    rows: int = 1000,
    prior: float = DEFAULT_PRIOR,
    shrinkage: float = DEFAULT_SHRINKAGE,
    cost: float = DEFAULT_COST,
    error_cost: float = DEFAULT_ERROR_COST,
) -> dict[str, Any]:
    """Симуляция по нескольким seed'ам: среднее и разброс итоговой оценки.

    Строки для всех оценщиков и режимов исследования генерируются одним и тем же
    потоком (общие случайные числа): различия в результате — это различия
    оценщиков, а не выборки.
    """
    results = []
    for seed in seeds:
        rng = random.Random(seed)
        rounds_rows = [make_rows(scenario, rng, rows) for _ in range(rounds)]
        result = run_policy(
            rounds_rows,
            estimator=estimator,
            explore=explore,
            q_true=scenario.q_true,
            prior=prior,
            shrinkage=shrinkage,
            cost=cost,
            error_cost=error_cost,
        )
        result["seed"] = seed
        results.append(result)
    q_values = [item["q_hat"] for item in results]
    biases = [item["bias"] for item in results]
    mean = sum(q_values) / len(q_values)
    variance = sum((value - mean) ** 2 for value in q_values) / len(q_values)
    bias_mean = sum(biases) / len(biases)
    bias_std = (
        sum((value - bias_mean) ** 2 for value in biases) / len(biases)
    ) ** 0.5
    return {
        "scenario": scenario.name,
        "description": scenario.description,
        "q_true": scenario.q_true,
        "prior": prior,
        "shrinkage": shrinkage,
        "estimator": estimator,
        "explore": explore,
        "q_hat_mean": mean,
        "q_hat_std": variance**0.5,
        "bias_mean": bias_mean,
        "bias_std": bias_std,
        "policy_rate": _mean(item["policy_rate"] for item in results),
        "calls_rate": _mean(item["calls_rate"] for item in results),
        "utility_rate": _mean(item["utility_rate"] for item in results),
        "oracle_rate": _mean(item["oracle_rate"] for item in results),
        "regret_rate": _mean(item["regret_rate"] for item in results),
        "effective_weight": _mean(item["weight_sum"] for item in results),
        "seeds": [item["seed"] for item in results],
        "runs": results,
    }


def _mean(values: Iterable[float]) -> float:
    items = list(values)
    return sum(items) / len(items) if items else 0.0


def report(
    scenarios: Iterable[str] | None = None,
    estimators: Iterable[str] = ("naive", "ips"),
    explores: Iterable[float] = (0.0, 0.05),
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """Полная сетка: сценарий × оценщик × доля исследования."""
    selected = scenarios or list(SCENARIOS)
    return [
        simulate(
            SCENARIOS[name],
            estimator=estimator,
            explore=explore,
            **kwargs,
        )
        for name in selected
        for estimator in estimators
        for explore in explores
    ]
