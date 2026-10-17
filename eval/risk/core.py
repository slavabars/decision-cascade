"""Честное сравнение маршрутизации: локально / порог уверенности / бюджет риска.

Считает на одном наборе качество локальных решений и компромисс «эскалация vs
качество принятых» для двух каскадов:

- пороговый — эскалация при ``confidence < threshold`` (текущее поведение);
- risk-aware — эскалация при ``P(ошибка) > risk_budget``.

Падение (fallback) считается идеальным: так измеряется *качество маршрутизации*,
а не сила внешней модели. Дополнительно считается AURC/eAURC — качество
ранжирования «быть правильным» по уверенности и по оценке риска.
"""
from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np
import torch

from eval.calibration.metrics import brier_score, ece
from eval.calibration.selective import aurc, coverage_at_risk, eaurc, risk_at_coverage
from src.sampler.questions import ChoiceSpec, NoulSpec, QuestionSpec
from src.schemas import DecisionResult


def labels_for(
    spec: QuestionSpec, examples: list[dict[str, Any]], name: str
) -> np.ndarray:
    raw = [ex["labels"][name] for ex in examples]
    if isinstance(spec, NoulSpec):
        return np.asarray(raw, dtype=float)
    if isinstance(spec, ChoiceSpec):
        index = {opt: i for i, opt in enumerate(spec.options)}
        return np.asarray([index[v] for v in raw], dtype=int)
    return np.asarray([int(v) for v in raw], dtype=int)


def is_correct(spec: QuestionSpec, decision: DecisionResult, label: Any) -> bool:
    if isinstance(spec, NoulSpec):
        return (decision.noul >= 0.5) == (float(label) >= 0.5)
    if isinstance(spec, ChoiceSpec):
        # ответ задан выбором: либо сам `choice`, либо индекс аргмакса распределения
        try:
            predicted = list(spec.options).index(decision.choice)
        except ValueError:
            probs = list(decision.probabilities.values())
            predicted = max(range(len(probs)), key=probs.__getitem__)
        return predicted == int(label)
    # ScoreSpec: как и в categorical_metrics — предсказанный уровень (argmax)
    probs = list(decision.probabilities.values())
    predicted = max(range(len(probs)), key=probs.__getitem__)
    return predicted == int(label)


def decisions_for_examples(
    heads, per_question: dict[str, torch.Tensor], n_examples: int
) -> dict[str, list[DecisionResult]]:
    """Решения по каждому вопросу для каждого примера — через ту же логику, что в сервисе."""
    names = list(heads.questions)
    collected: dict[str, list[DecisionResult]] = {name: [] for name in names}
    for i in range(n_examples):
        row = torch.stack([per_question[name][i] for name in names])
        results = heads.evaluate(row)
        for name in names:
            collected[name].append(results[name])
    return collected


def flatten(
    specs: dict[str, QuestionSpec],
    decisions: dict[str, list[DecisionResult]],
    examples: list[dict[str, Any]],
) -> tuple[list[DecisionResult], np.ndarray]:
    """Плоский список решений и их корректности по всем вопросам."""
    flat_decisions: list[DecisionResult] = []
    flat_correct: list[float] = []
    for name, spec in specs.items():
        labels = labels_for(spec, examples, name)
        for decision, label in zip(decisions[name], labels):
            flat_decisions.append(decision)
            flat_correct.append(float(is_correct(spec, decision, label)))
    return flat_decisions, np.asarray(flat_correct, dtype=float)


def threshold_escalation(decisions: list[DecisionResult], threshold: float) -> np.ndarray:
    return np.asarray([d.confidence < threshold for d in decisions], dtype=bool)


def risk_escalation(decisions: list[DecisionResult], model, budget: float) -> np.ndarray:
    return np.asarray(
        [model.should_escalate(d, budget) for d in decisions], dtype=bool
    )


def cascade_summary(correct: np.ndarray, escalate: np.ndarray) -> dict[str, float]:
    """Метрики каскада при идеальном fallback."""
    correct = np.asarray(correct, dtype=float).reshape(-1)
    escalate = np.asarray(escalate, dtype=bool).reshape(-1)
    n = correct.size
    accepted = ~escalate
    accepted_accuracy = float(correct[accepted].mean()) if accepted.any() else float("nan")
    end2end = float((correct[accepted].sum() + escalate.sum()) / n)
    return {
        "escalation_rate": float(escalate.mean()),
        "accepted_rate": float(accepted.mean()),
        "accepted_accuracy": accepted_accuracy,
        "accepted_risk": 1.0 - accepted_accuracy if accepted.any() else float("nan"),
        "end2end_accuracy": end2end,
    }


def ranker_report(scores: np.ndarray, correct: np.ndarray) -> dict[str, float]:
    """AURC/eAURC, покрытие при бюджете риска 5% и 10%, калибровка ``P(ошибка)``.

    ``scores`` — балл «решение верно», поэтому ECE/Brier считаются на
    вероятности ошибки ``1 - score`` против факта ``1 - correct``.
    """
    error_prob = 1.0 - np.asarray(scores, dtype=float)
    errors = 1.0 - np.asarray(correct, dtype=float)
    return {
        "aurc": aurc(scores, correct),
        "eaurc": eaurc(scores, correct),
        "coverage_at_5pct_risk": coverage_at_risk(scores, correct, 0.05),
        "coverage_at_10pct_risk": coverage_at_risk(scores, correct, 0.10),
        "ece": float(ece(error_prob, errors)),
        "brier": float(brier_score(error_prob, errors)),
    }


#: Сколько bootstrap-выборок по умолчанию. Дёшево (numpy), но сглаживает
#: хвост перцентилей: 2000 хватает для 95% CI в двух знаках.
BOOTSTRAP_DRAWS = 2000


def bootstrap_mean_ci(
    values: Any,
    *,
    n_boot: int = BOOTSTRAP_DRAWS,
    level: float = 0.95,
    seed: int = 0,
) -> dict[str, float]:
    """Перцентильный bootstrap-CI **среднего** для 1-D выборки.

    Перевыборка единиц измерения (пар «seed×вопрос» или тестовых строк) с
    возвращением. Даёт то, чего не может постоянный допуск: интервал, внутри
    которого среднее «плавает» из-за конечности замера.
    """
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {
            "mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0
        }
    rng = np.random.default_rng(seed)
    index = rng.integers(0, array.size, size=(n_boot, array.size))
    means = array[index].mean(axis=1)
    tail = (1.0 - level) / 2.0
    lo, hi = np.quantile(means, [tail, 1.0 - tail])
    return {
        "mean": float(array.mean()),
        "lo": float(lo),
        "hi": float(hi),
        "n": int(array.size),
    }


def bootstrap_ranker_ci(
    scores: np.ndarray,
    correct: np.ndarray,
    *,
    n_boot: int = BOOTSTRAP_DRAWS,
    level: float = 0.95,
    seed: int = 0,
) -> dict[str, dict[str, float]]:
    """Bootstrap-CI метрик ранжирования по **тестовым строкам**.

    Единица перевыборки — строка теста: именно её не хватает, чтобы отличить
    «эффект» от «конечной выборки». Числа в CI считаются тем же
    ``ranker_report``, поэтому интервал согласован с точечной оценкой.
    """
    scores = np.asarray(scores, dtype=float)
    correct = np.asarray(correct, dtype=float)
    if scores.size == 0:
        return {}
    rng = np.random.default_rng(seed)
    index = rng.integers(0, scores.size, size=(n_boot, scores.size))
    accumulated: dict[str, list[float]] = {}
    for draw in range(n_boot):
        pick = index[draw]
        report = ranker_report(scores[pick], correct[pick])
        for key, value in report.items():
            accumulated.setdefault(key, []).append(value)
    tail = (1.0 - level) / 2.0
    point = ranker_report(scores, correct)
    ci: dict[str, dict[str, float]] = {}
    for key, values in accumulated.items():
        lo, hi = np.quantile(values, [tail, 1.0 - tail])
        ci[key] = {"value": float(point[key]), "lo": float(lo), "hi": float(hi)}
    return ci


def paired_differences(
    rows: list[dict[str, Any]],
    metrics: tuple[GateMetric, ...],
    *,
    unit: tuple[str, ...] = ("seed", "question"),
    baseline: str = "confidence",
    candidate: str = "production",
) -> dict[str, list[float]]:
    """Разница ``candidate − baseline`` по парам ``unit`` в направлении «хуже».

    Для метрик «больше лучше» знак переворачивается, поэтому положительное
    значение **всегда** означает «кандидат хуже baseline». Возвращает по списку
    различий на метрику — их и подают в `bootstrap_mean_ci`.
    """
    join = (*unit, "variant")
    index = {tuple(row.get(key) for key in join): row for row in rows}
    differences: dict[str, list[float]] = {metric.key: [] for metric in metrics}
    for key, row in index.items():
        if key[-1] != candidate:
            continue
        base = index.get((*key[:-1], baseline))
        if base is None:
            continue
        for metric in metrics:
            if metric.key not in row or metric.key not in base:
                continue
            direction = 1.0 if metric.lower_better else -1.0
            differences[metric.key].append(
                direction * (float(row[metric.key]) - float(base[metric.key]))
            )
    return differences


def stability_report(
    differences: dict[str, list[float]],
    *,
    n_boot: int = BOOTSTRAP_DRAWS,
    level: float = 0.95,
    seed: int = 0,
) -> dict[str, dict[str, float]]:
    """CI среднего различия + доля пар, где кандидат хуже baseline.

    ``significant`` — интервал не накрывает ноль: разницу нельзя объяснить
    конечностью замера. Это оценка для решения «принимать только лучшее»,
    отдельная от жёсткого по-точечного гейта.
    """
    report: dict[str, dict[str, float]] = {}
    for key, values in differences.items():
        if not values:
            continue
        ci = bootstrap_mean_ci(values, n_boot=n_boot, level=level, seed=seed)
        ci["share_worse"] = float(np.mean(np.asarray(values) > 0.0))
        ci["significant"] = bool(ci["lo"] > 0.0 or ci["hi"] < 0.0)
        report[key] = ci
    return report


DEFAULT_COVERAGES = (0.5, 0.7, 0.9)

#: Допуск только на шумовые колебания. Регрессия сильнее — отказ от варианта.
#: Для AURC/eAURC/покрытия берём по решению проекта жёсткие 0.0005: регрессия
#: не прощается, допуск съедает только численный шум.
GATE_TOL = 0.0005

#: Допуски калибровки — по **измеренному** разбросу. Их можно ставить жёстко
#: только там, где метрика разрешима: в пуле вопросов (~3000 тестовых строк,
#: RouterBench/hash) измеренный seed-σ разницы `production − confidence` —
#: 0.024 у ECE и 0.004 у Brier, а наблюдаемые эффекты 0.04–0.09. Берём ≈1σ:
#: порог намеренно строгий, ложный отказ безопасен (фолбэк на `confidence`).
#: На **одном** новом вопросе ECE так не работает (см. `GATE_LOO_METRICS`).
GATE_TOL_ECE = 0.02
GATE_TOL_BRIER = 0.01


class GateMetric(NamedTuple):
    """Метрика гейта: ключ, направление («меньше лучше») и допуск на шум."""

    key: str
    lower_better: bool
    tol: float = GATE_TOL


#: AURC/eAURC/покрытие/точность — рабочие SLO риск-каскада, калибровка — честность
#: `P(ошибка)`, по которой каскад сравнивает с бюджетом.
_GATE_RANKING = (
    GateMetric("aurc", True),
    GateMetric("eaurc", True),
    GateMetric("coverage_at_5pct_risk", False),
)
_GATE_CALIBRATION = (
    GateMetric("ece", True, GATE_TOL_ECE),
    GateMetric("brier", True, GATE_TOL_BRIER),
)

#: Гейт **пула** вопросов: in-dist A/B (`--shots` в точке «все») — тут калибровка
#: разрешима, поэтому входит в критерии.
GATE_AB_METRICS = (
    *_GATE_RANKING,
    GateMetric("acc_at_0.7_coverage", False),
    *_GATE_CALIBRATION,
)
#: Гейт сертификата `eval/risk/run.py`: тоже пул (перенос считается идеальным),
#: калибровка входит.
GATE_RANKER_METRICS = (
    GateMetric("aurc", True),
    GateMetric("eaurc", True),
    GateMetric("coverage_at_5pct_risk", False),
    GateMetric("coverage_at_10pct_risk", False),
    *_GATE_CALIBRATION,
)
#: Гейт **одного нового вопроса** (LOO). Калибровки здесь нет намеренно:
#: измеренный разброс разницы `production − confidence` по seed'ам — σ_ECE
#: 0.03–0.07 (и всего ~3 независимых прогона, порядок примеров при n=0 не
#: добавляет информации), то есть любой осмысленный допуск отвергал бы на шуме.
#: Замеры показали, что «калибровочные» отказы Шага 1 были артефактом одного
#: seed'а: усреднение по трём даёт +0.007 ± 0.030 (эффекта нет). Калибровку в
#: этот гейт вернёт Шаг 3 по bootstrap-CI, а не по константному допуску.
GATE_LOO_METRICS = (
    GateMetric("loo_aurc", True),
    GateMetric("loo_eaurc", True),
    GateMetric("loo_coverage_at_5pct_risk", False),
    GateMetric("loo_acc_at_0.7", False),
)
#: Гейт кривой few-shot: срез идёт по вопросу, поэтому как у LOO — без калибровки.
GATE_SHOTS_METRICS = (
    *_GATE_RANKING,
    GateMetric("acc_at_0.7_coverage", False),
)


def gate_tolerances(metrics: tuple[GateMetric, ...]) -> dict[str, float]:
    """Допуски метрик в виде словаря — чтобы отчёт/сертификат был самодостаточным."""
    return {metric.key: metric.tol for metric in metrics}


def gate_risk_vs_confidence(
    baseline: dict[str, float],
    candidate: dict[str, float],
    metrics: tuple[GateMetric, ...] = GATE_AB_METRICS,
    *,
    tol: float | None = None,
) -> tuple[bool, list[str]]:
    """Hard gate: кандидат принимается **только** если не хуже baseline.

    Возвращает ``(accepted, reasons)``: ``accepted`` истинен лишь когда по
    **всем** перечисленным метрикам кандидат не хуже baseline (допуск поедает
    только шум). Любая регрессия — причина отказа, тогда на выход идёт лучший
    baseline (по умолчанию `confidence`).

    Параметры:
    - ``baseline`` / ``candidate`` — словари метрик;
    - ``metrics`` — кортеж ``GateMetric`` (ключ, меньше_лучше, допуск);
    - ``tol`` — переопределить допуск **всех** метрик одним числом
      (по умолчанию у каждой метрики свой, см. ``GATE_TOL*``).
    """
    worse: list[str] = []
    for metric in metrics:
        key = metric.key
        if key not in baseline or key not in candidate:
            continue
        limit = metric.tol if tol is None else float(tol)
        base, cand = float(baseline[key]), float(candidate[key])
        bad = cand > base + limit if metric.lower_better else cand < base - limit
        if bad:
            relation = ">" if metric.lower_better else "<"
            worse.append(f"{key}: {cand:.4f} {relation} {base:.4f}±{limit:g}")
    return len(worse) == 0, worse


def gate_excess(
    baseline: dict[str, float],
    candidate: dict[str, float],
    metrics: tuple[GateMetric, ...] = GATE_AB_METRICS,
    *,
    tol: float | None = None,
) -> dict[str, float]:
    """Насколько каждая метрика кандидата хуже baseline **сверх** допуска.

    Положительное значение — регрессия за пределами допуска (именно она
    отвергает вариант), нуль и меньше — «в пределах шума». Пишется в сертификат
    (``gate["excess"]``), чтобы прод мог пересчитать вердикт под своим допуском
    (``DECISION_RISK_GATE_DELTA``, см. ``src.risk.gate_verdict``) без
    переобучения модели.
    """
    excess: dict[str, float] = {}
    for metric in metrics:
        key = metric.key
        if key not in baseline or key not in candidate:
            continue
        limit = metric.tol if tol is None else float(tol)
        base, cand = float(baseline[key]), float(candidate[key])
        margin = cand - base if metric.lower_better else base - cand
        excess[key] = margin - limit
    return excess


def selective_accuracy(
    scores: np.ndarray, correct: np.ndarray, coverages=DEFAULT_COVERAGES
) -> dict[str, float]:
    """Точность среди принимаемых решений при заданной доле принятия.

    Честное сравнение при *равных затратах*: какой подмножество принимаемых
    решений точнее, если во внешнюю модель уходит одинаковая доля запросов.
    """
    return {
        f"{c:.1f}": 1.0 - risk_at_coverage(scores, correct, c)
        for c in coverages
    }
