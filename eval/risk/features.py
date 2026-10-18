"""A/B-замер состава признаков оценщика риска.

Признаки решения считает `src.risk.decision_features` — это то, что реально
подключается в сервисе. Этот CLI отвечает на вопрос: **какие** из них дают
сигнал об ошибке, добавляясь к трём базовым (`confidence, entropy, margin`).

Проверяются:

- `base(3)` — три базовых признака: прежняя модель (до расширения);
- `random` — случайный балл («без навыка»): его AURC ≈ базовая частота ошибок,
  то есть граница «хуже случайного»;

С `--question` добавляются варианты с **явным свободным членом на вопрос**
(one-hot): `prod+question` — шесть признаков плюс по смещению на вопрос,
`no_n_classes+question` — то же без признака `n_classes`. Так проверяется, можно
ли заменить суррогат «номера вопроса» (`n_classes` «узнаёт» вопрос по числу
исходов) на явный свободный член.
- `+log_odds_gap`, `+n_classes`, `+noul_logit` — base плюс **один** новый
  признак (видно, кто тянет выигрыш);
- `production` — все признаки из `FEATURE_NAMES`, то есть ровно то, что
  обучает `training.recipe.risk` и что проверяется в `eval.risk.run`
  (числа должны совпадать с `risk_model` оттуда);
- `+option_similarity` — production плюс признак, которому нужны
  **внутренности головы** (средняя попарная косинусная близость направлений
  вариантов) и потому поле в схеме `DecisionResult`.

С `--state` добавляются варианты с **состоянием энкодера** (пулированный вектор
промпта, размерность `--dim`): `state[C]` — только состояние, `prod+state[C]` —
шесть признаков плюс состояние. Состояние стандартизуется по валидации, а сила L2
(`C`) меньше, чем у шести признаков, иначе 768 измерений на 400 строках валидации
переобучаются; можно задать несколько `--state-C`, чтобы увидеть чувствительность.
Это проверка гипотезы «признаки из головы — узкое место, нужен сигнал из
состояния», а не production-путь: в сервисе оценщик риска пока видит только
`DecisionResult`.

С `--qmean` (Шаг 5) добавляются варианты со **средним пулированным вектором
вопроса**: `qmean[C]` — только он, `prod+qmean[C]` — шесть признаков плюс он.
Вектор усредняется по **валидации** вопроса и нормируется (в тест не подглядываем:
в сервисе так же усредняются промпты нового вопроса без меток), поэтому внутри
вопроса он константа — тот же класс признака, что `n_classes`/one-hot, но
необучаемый и «мягкий». Это проверка гипотезы «жёсткий `n_classes` — суррогат,
который можно заменить/дополнить мягким prior'ом вопроса».

С `--ablate` (Шаг 5) добавляются варианты `production` **минус один** новый
признак (`no_n_classes`, `no_log_odds_gap`, `no_noul_logit`). Нужны, чтобы
отделить вклад признака от вклада остальных: замер Шага 3 показал, что именно
`noul_logit` значимо хуже `confidence` на новом вопросе, и абляция проверяет это
напрямую (а не через «`+` к базовым»).

С `--C-grid` (Шаг 5а) сетка по силе L2: варианты `production[C]` делят с `production`
ту же матрицу признаков, отличаясь только регуляризацией. Сетка по `C` — один
прогон, а не прогон на значение, потому что кодирование и обучение голов
общие; это измерение того, не «чинит» ли перенос обычная регуляризация.

С `--shots` считается **кривая дообучения на новом вопросе**: риск-модель учится
на валидации остальных вопросов плюс `n` случайных строк валидации самого вопроса,
а оценивается на его тесте. `n = 0` — это перенос без своих данных (`--loo`), `n` =
вся валидация вопроса — обучение на всех вопросах (верхняя граница, `in-dist`).
Кривая отвечает на продуктовый вопрос: сколько размеченных примеров нового
вопроса нужно, чтобы бюджет ошибок на нём заработал.

Вариант `+option_similarity` измеряется, но в production не подключён: его
выигрыш меньше, чем цена изменения схемы и API.

Модель учится на валидации и оценивается на тесте — как в `eval.risk.run`.
Балл ранжирования — `1 - P(ошибка)`, то есть «чем больше, тем лучше»: в
`eval.risk.run` именно так подаётся `confidence`, поэтому числа сопоставимы.
Признаки не масштабируются (как в `training.recipe.risk`); `--scale` включает
стандартизацию по статистике валидации — для проверки, что вывод не зависит от
масштаба (проверено: не зависит).

Пример:

    python -m eval.risk.features --specs routerbench \
        --encoder openai-community/gpt2 --dim 768 --train-steps 500 --seeds 0 1 2
"""
from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from eval.benchmark.core import TextEncoder, encode_examples, train_benchmark
from eval.calibration.metrics import brier_score, ece
from eval.calibration.selective import aurc, coverage_at_risk, eaurc, risk_at_coverage
from eval.report.run import _encoder, _load_splits, _specs
from eval.risk.core import (
    GATE_AB_METRICS,
    GATE_LOO_METRICS,
    GATE_SHOTS_METRICS,
    GATE_TOL,
    GateMetric,
    decisions_for_examples,
    flatten,
    gate_risk_vs_confidence,
    gate_tolerances,
    paired_differences,
    stability_report,
)
from src.decision_head.heads import OptionScoringChoiceHead
from src.risk import FEATURE_NAMES, RISK_C_DEFAULT, features_matrix
from src.sampler.questions import QuestionSpec
from src.schemas import DecisionResult

#: Признаки, добавленные к трём базовым (уже входят в ``FEATURE_NAMES``).
EXTRA_FEATURE_NAMES = tuple(FEATURE_NAMES[3:])
#: Признак, требующий внутренностей головы (в production не подключён).
HEAD_FEATURE_NAMES = ("option_similarity",)
#: Вариант со всем, что подключено в сервисе.
PRODUCTION_VARIANT = "production"
#: Число базовых признаков (до расширения) — префикс ``FEATURE_NAMES``.
BASE_WIDTH = 3


def option_directions(head) -> Any | None:
    """Направления вариантов головы; ``None`` — если вариантов нет (`noul`)."""
    if isinstance(head, OptionScoringChoiceHead):
        return head.option_proj(head.option_embeddings).detach()
    weight = getattr(head, "linear", None)
    return None if weight is None else weight.weight.detach()


def mean_pairwise_cosine(vectors) -> float:
    """Средняя косинусная близость попарно разных вариантов (0 — если их < 2)."""
    if vectors is None or vectors.shape[0] < 2:
        return 0.0
    unit = vectors / vectors.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    gram = (unit @ unit.t()).numpy()
    return float(gram[np.triu_indices(gram.shape[0], k=1)].mean())


def head_features(
    heads,
    specs: dict[str, QuestionSpec],
    decisions: dict[str, list[DecisionResult]],
) -> np.ndarray:
    """``[n, len(HEAD_FEATURE_NAMES)]``: признак головы, повторённый по строкам.

    Порядок строк — question-major, как в `eval.risk.core.flatten`.
    """
    columns = [
        np.full(
            len(decisions[question]),
            mean_pairwise_cosine(option_directions(heads.heads[question])),
        ).reshape(-1, 1)
        for question in specs
    ]
    return np.vstack(columns)


def _scaled(
    train: np.ndarray, test: np.ndarray, scale: bool
) -> tuple[np.ndarray, np.ndarray]:
    """Стандартизация колонок по статистике ``train`` (не подглядываем в test)."""
    if not scale:
        return train, test
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    std[std < 1e-12] = 1.0
    return (train - mean) / std, (test - mean) / std


def _fit_logistic(
    features: np.ndarray, correct: np.ndarray, *, seed: int = 0,
    C: float = RISK_C_DEFAULT,
) -> tuple[np.ndarray, float]:
    """Логистическая регрессия ``признаки -> P(ошибка)``.

    Тот же вызов, что в `training.recipe.risk`: замер обязан повторять
    production-обучение, иначе числа ничего не значат про сервис.
    """
    from sklearn.linear_model import LogisticRegression

    features = np.asarray(features, dtype=float)
    errors = 1 - np.asarray(correct, dtype=int).reshape(-1)
    if len(np.unique(errors)) < 2:
        eps = 1e-6
        p = float(np.clip(errors.mean(), eps, 1.0 - eps))
        return np.zeros(features.shape[1]), float(np.log(p / (1.0 - p)))
    clf = LogisticRegression(C=C, random_state=seed, max_iter=1000)
    clf.fit(features, errors)
    return clf.coef_[0], float(clf.intercept_[0])


def calibration_report(scores: np.ndarray, correct: np.ndarray) -> dict[str, float]:
    """Калибровка оценки ошибки: ECE и Brier score (меньше лучше).

    ``scores`` — балл «решение верно» (чем больше, тем увереннее). Модель риска
    предсказывает ``P(ошибка) = 1 - score``, поэтому обе метрики считаются на
    вероятности ошибки против фактической ошибки: ECE — |средний P(ошибка) −
    доля ошибок| по бинам, Brier — средний квадрат той же разницы.
    """
    error_prob = 1.0 - np.asarray(scores, dtype=float)
    errors = 1.0 - np.asarray(correct, dtype=float)
    return {
        "ece": float(ece(error_prob, errors)),
        "brier": float(brier_score(error_prob, errors)),
    }


def risk_report(scores: np.ndarray, correct: np.ndarray) -> dict[str, float]:
    """AURC/eAURC, покрытие при бюджете 5%, точность при покрытии 0.7, калибровка."""
    return {
        "aurc": aurc(scores, correct),
        "eaurc": eaurc(scores, correct),
        "coverage_at_5pct_risk": coverage_at_risk(scores, correct, 0.05),
        "acc_at_0.7_coverage": 1.0 - risk_at_coverage(scores, correct, 0.7),
        **calibration_report(scores, correct),
    }


def feature_variants(
    heads,
    specs: dict[str, QuestionSpec],
    val_decisions: dict[str, list[DecisionResult]],
    test_decisions: dict[str, list[DecisionResult]],
    val_matrix: np.ndarray,
    test_matrix: np.ndarray,
    *_,
    scale: bool = False,
    question: bool = False,
    ablate: bool = False,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Матрицы признаков по вариантам состава (train — валидация, test — тест)."""
    columns = list(range(BASE_WIDTH))
    variants: dict[str, tuple[np.ndarray, np.ndarray]] = {
        # «сырая» уверенность — нижняя точка отсчёта: ``base(3)`` берёт
        # ``d.confidence``, как её считают пороговый каскад и ``eval.risk.run``
        "confidence": (
            np.asarray([[d.confidence] for d in _flat(val_decisions)], dtype=float),
            np.asarray([[d.confidence] for d in _flat(test_decisions)], dtype=float),
        ),
        "base(3)": (val_matrix[:, columns], test_matrix[:, columns]),
    }
    # «без навыка»: случайный балл. Его AURC ≈ базовая частота ошибок, поэтому
    # он отделяет «модель вообще не работает» от «модель работает хуже случайной».
    rng = np.random.default_rng(0)
    variants["random"] = (
        rng.random(len(_flat(val_decisions))).reshape(-1, 1),
        rng.random(len(_flat(test_decisions))).reshape(-1, 1),
    )
    for offset in range(BASE_WIDTH, len(FEATURE_NAMES)):
        selected = columns + [offset]
        variants[f"+{FEATURE_NAMES[offset]}"] = (
            val_matrix[:, selected],
            test_matrix[:, selected],
        )
    variants[PRODUCTION_VARIANT] = (val_matrix, test_matrix)
    if ablate:
        # «production минус один новый признак»: нужен, чтобы отделить вклад
        # каждого признака от вклада остальных (Шаг 5, абляция `noul_logit`).
        for offset in range(BASE_WIDTH, len(FEATURE_NAMES)):
            name = FEATURE_NAMES[offset]
            kept = [column for column in range(len(FEATURE_NAMES)) if column != offset]
            variants[f"no_{name}"] = (val_matrix[:, kept], test_matrix[:, kept])
    head_val, head_test = _scaled(
        head_features(heads, specs, val_decisions),
        head_features(heads, specs, test_decisions),
        scale,
    )
    variants["+option_similarity"] = (
        np.hstack([val_matrix, head_val]),
        np.hstack([test_matrix, head_test]),
    )
    if question:
        val_question = question_onehot(specs, val_decisions)
        test_question = question_onehot(specs, test_decisions)
        variants["prod+question"] = (
            np.hstack([val_matrix, val_question]),
            np.hstack([test_matrix, test_question]),
        )
        without_n_classes = [
            column
            for column in range(len(FEATURE_NAMES))
            if FEATURE_NAMES[column] != "n_classes"
        ]
        variants["no_n_classes+question"] = (
            np.hstack([val_matrix[:, without_n_classes], val_question]),
            np.hstack([test_matrix[:, without_n_classes], test_question]),
        )
    return variants


def _flat(decisions: dict[str, list[DecisionResult]]) -> list[DecisionResult]:
    return [decision for group in decisions.values() for decision in group]


def question_onehot(
    specs: dict[str, QuestionSpec],
    decisions: dict[str, list[DecisionResult]],
) -> np.ndarray:
    """One-hot вопроса — явный свободный член на вопрос.

    Строки идут question-major (как у `_groups`). На отложенном вопросе
    (``--loo``) его колонка в обучении целиком нулевая, поэтому модель работает
    без свободного члена под этот вопрос — в отличие от ``n_classes``, который
    «узнаёт» вопрос по числу исходов и тащит на него чужой коэффициент.
    """
    index = {name: position for position, name in enumerate(specs)}
    matrix = np.zeros((sum(len(group) for group in decisions.values()), len(specs)))
    row = 0
    for name, group in decisions.items():
        for _decision in group:
            matrix[row, index[name]] = 1.0
            row += 1
    return matrix


def _c_label(c: float) -> str:
    return f"[{c:g}]"


def with_state_variants(
    variants: dict[str, tuple[np.ndarray, np.ndarray]],
    states: tuple[np.ndarray, np.ndarray],
    cs: list[float],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Добавляет варианты с состоянием энкодера: ``state[C]`` и ``prod+state[C]``.

    Состояние стандартизуется по статистике валидации (иначе масштаб колонок
    разъезжается и L2 штрафует их неравномерно).
    """
    state_val, state_test = _scaled(states[0], states[1], True)
    prod_val, prod_test = variants[PRODUCTION_VARIANT]
    out = dict(variants)
    for c in cs:
        label = _c_label(c)
        out[f"state{label}"] = (state_val, state_test)
        out[f"prod+state{label}"] = (
            np.hstack([prod_val, state_val]),
            np.hstack([prod_test, state_test]),
        )
    return out


def _qmean_vectors(val_features: dict[str, Any]) -> dict[str, np.ndarray]:
    """Нормированный средний пулированный вектор вопроса.

    Среднее берётся по **валидации** вопроса, поэтому в тест не подглядываем: в
    сервисе ровно так же усредняются промпты нового вопроса, метки которого
    неизвестны. Вектор — константа вопроса, то есть тот же класс, что
    ``n_classes``/one-hot, но необучаемый и «мягкий».
    """
    vectors: dict[str, np.ndarray] = {}
    for question, states in val_features.items():
        mean = states.mean(dim=0)
        norm = float(mean.norm().clamp(min=1e-12))
        vectors[question] = (mean / norm).detach().cpu().numpy().astype(float)
    return vectors


def qmean_matrix(
    vectors: dict[str, np.ndarray],
    features: dict[str, Any],
    specs: dict[str, QuestionSpec],
) -> np.ndarray:
    """Развернуть векторы вопросов по строкам (question-major, как `flatten`)."""
    return np.concatenate(
        [
            np.tile(vectors[question], (features[question].shape[0], 1))
            for question in specs
        ]
    )


def with_qmean_variants(
    variants: dict[str, tuple[np.ndarray, np.ndarray]],
    qmeans: tuple[np.ndarray, np.ndarray],
    cs: list[float],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Добавляет варианты со средним вектором вопроса: ``qmean[C]`` и ``prod+qmean[C]``.

    Колонки **не** стандартизуются по строкам: вектор постоянен внутри вопроса,
    и стандартизация обнулила бы его. Масштаб задаёт нормировка вектора, а силу
    L2 подбирает ``--qmean-C`` (как у состояния энкодера).
    """
    qmean_val, qmean_test = qmeans
    prod_val, prod_test = variants[PRODUCTION_VARIANT]
    out = dict(variants)
    for c in cs:
        label = _c_label(c)
        out[f"qmean{label}"] = (qmean_val, qmean_test)
        out[f"prod+qmean{label}"] = (
            np.hstack([prod_val, qmean_val]),
            np.hstack([prod_test, qmean_test]),
        )
    return out


def _groups(specs: dict[str, QuestionSpec], decisions) -> list[str]:
    """Имя вопроса по каждой строке — в том же порядке, что у `flatten`."""
    return [name for name in specs for _ in decisions[name]]


@dataclass
class _Prepared:
    """Матрицы признаков и корректность по строкам, посчитанные один раз."""

    #: вариант состава признаков -> (матрица валидации, матрица теста)
    variants: dict[str, tuple[np.ndarray, np.ndarray]]
    #: (корректность валидации, корректность теста)
    correct: tuple[np.ndarray, np.ndarray]
    #: (имя вопроса по строкам валидации, имя вопроса по строкам теста)
    groups: tuple[list[str], list[str]]
    #: порядок вопросов
    questions: list[str]
    #: вариант -> сила L2 (для вариантов с состоянием она своя)
    cs: dict[str, float] = field(default_factory=dict)


def prepare(
    heads,
    encoder: TextEncoder,
    specs: dict[str, QuestionSpec],
    val: list[dict[str, Any]],
    test: list[dict[str, Any]],
    *,
    scale: bool = False,
    with_state: bool = False,
    state_cs: list[float] | None = None,
    with_question: bool = False,
    ablate: bool = False,
    with_qmean: bool = False,
    qmean_cs: list[float] | None = None,
    c_grid: list[float] | None = None,
) -> _Prepared:
    """Один раз считает решения, признаки и корректность для A/B и LOO.

    ``with_state`` добавляет варианты с состоянием энкодера (см.
    `with_state_variants`), ``state_cs`` — их силы L2, ``with_question`` —
    варианты с явным свободным членом на вопрос (см. `question_onehot`),
    ``ablate`` — варианты ``production`` без одного нового признака, ``with_qmean``
    — варианты со средним вектором вопроса (см. `with_qmean_variants`).
    ``c_grid`` добавляет ``production[C]`` с той же матрицей признаков, но своей
    силой L2: признаки считаются один раз, поэтому сетка по ``C`` не требует
    повторного кодирования (см. Шаг 5а).
    """
    val_features = encode_examples(specs, val, encoder)
    val_decisions = decisions_for_examples(heads, val_features, len(val))
    val_flat, val_correct = flatten(specs, val_decisions, val)
    test_features = encode_examples(specs, test, encoder)
    test_decisions = decisions_for_examples(heads, test_features, len(test))
    test_flat, test_correct = flatten(specs, test_decisions, test)

    val_matrix, test_matrix = _scaled(
        features_matrix(val_flat), features_matrix(test_flat), scale
    )
    variants = feature_variants(
        heads, specs, val_decisions, test_decisions, val_matrix, test_matrix,
        scale=scale, question=with_question, ablate=ablate,
    )
    cs: dict[str, float] = {}
    if c_grid:
        # та же матрица признаков, своя L2: сетка по C без повторного кодирования
        production = variants[PRODUCTION_VARIANT]
        for c in c_grid:
            label = _c_label(c)
            variants[f"{PRODUCTION_VARIANT}{label}"] = production
            cs[f"{PRODUCTION_VARIANT}{label}"] = c
    if with_state:
        state_cs = list(state_cs or [0.05])
        # порядок строк — question-major, как у ``flatten``
        states = (
            _state_matrix(val_features, specs),
            _state_matrix(test_features, specs),
        )
        variants = with_state_variants(variants, states, state_cs)
        for c in state_cs:
            label = _c_label(c)
            cs[f"state{label}"] = c
            cs[f"prod+state{label}"] = c
    if with_qmean:
        qmean_cs = list(qmean_cs or [0.05])
        vectors = _qmean_vectors(val_features)
        qmeans = (
            qmean_matrix(vectors, val_features, specs),
            qmean_matrix(vectors, test_features, specs),
        )
        variants = with_qmean_variants(variants, qmeans, qmean_cs)
        for c in qmean_cs:
            label = _c_label(c)
            cs[f"qmean{label}"] = c
            cs[f"prod+qmean{label}"] = c
    return _Prepared(
        variants=variants,
        correct=(val_correct, test_correct),
        groups=(_groups(specs, val_decisions), _groups(specs, test_decisions)),
        questions=list(specs),
        cs=cs,
    )


def _state_matrix(
    features: dict[str, Any], specs: dict[str, QuestionSpec]
) -> np.ndarray:
    """Пулированные состояния по строкам в порядке question-major."""
    return np.concatenate(
        [features[question].detach().cpu().numpy().astype(float) for question in specs]
    )


def _scores(
    train_matrix: np.ndarray,
    eval_matrix: np.ndarray,
    correct: np.ndarray,
    *,
    seed: int,
    C: float,
) -> np.ndarray:
    """Балл «быть правильным» на ``eval_matrix``; модель учится на ``train_matrix``."""
    weights, bias = _fit_logistic(train_matrix, correct, seed=seed, C=C)
    # модель предсказывает P(ошибка); ранжировщику нужен балл «чем больше, тем
    # лучше» — как в ``eval.risk.run``
    error = 1.0 / (1.0 + np.exp(-(eval_matrix @ weights + bias)))
    return 1.0 - error


def _c_for(prepared: _Prepared, name: str, default: float) -> float:
    """Сила L2 для варианта: у вариантов с состоянием она своя."""
    return prepared.cs.get(name, default)


def run_ab(
    prepared: _Prepared, *, seed: int = 0, C: float = RISK_C_DEFAULT
) -> list[dict[str, float]]:
    """Подгоняет риск-модели на валидации и считает метрики на тесте."""
    _, test_correct = prepared.correct
    rows = []
    for name, (train_matrix, eval_matrix) in prepared.variants.items():
        scores = _scores(
            train_matrix, eval_matrix, prepared.correct[0], seed=seed,
            C=_c_for(prepared, name, C),
        )
        rows.append({"variant": name, "seed": seed, **risk_report(scores, test_correct)})
    return rows


def run_loo(
    prepared: _Prepared, *, seed: int = 0, C: float = RISK_C_DEFAULT
) -> list[dict[str, Any]]:
    """Leave-one-question-out: обучать без одного вопроса, замерять на нём.

    В обычном A/B валидация и тест делят один и тот же набор вопросов, поэтому
    признак-константа вопроса (``n_classes``) работает как «я знаю, какой это
    вопрос»: модель просто подбирает точку срабатывания под вопрос. Здесь
    риск-модель обучается на всех вопросах, **кроме одного**, и оценивается на
    отложенном. Это несмещённая оценка того, что переносится на новый вопрос.

    Рядом считается ``indist`` — на том же подмножестве, но у модели, обученной
    на всех вопросах. Разрыв ``indist - loo`` и есть вклад «знания вопроса».
    """
    val_correct, test_correct = prepared.correct
    val_groups = np.asarray(prepared.groups[0])
    test_groups = np.asarray(prepared.groups[1])

    rows: list[dict[str, Any]] = []
    for name, (val_matrix, test_matrix) in prepared.variants.items():
        c = _c_for(prepared, name, C)
        indist = _scores(val_matrix, test_matrix, val_correct, seed=seed, C=c)
        for question in prepared.questions:
            train_rows = val_groups != question
            test_rows = test_groups == question
            loo = _scores(
                val_matrix[train_rows],
                test_matrix[test_rows],
                val_correct[train_rows],
                seed=seed,
                C=c,
            )
            subset_correct = test_correct[test_rows]
            rows.append(
                {
                    "variant": name,
                    "question": question,
                    "seed": seed,
                    "n_val": int(train_rows.sum()),
                    "n_test": int(test_rows.sum()),
                    "loo_aurc": aurc(loo, subset_correct),
                    "loo_eaurc": eaurc(loo, subset_correct),
                    "loo_coverage_at_5pct_risk": coverage_at_risk(
                        loo, subset_correct, 0.05
                    ),
                    "loo_acc_at_0.7": 1.0
                    - risk_at_coverage(loo, subset_correct, 0.7),
                    **{
                        f"loo_{key}": value
                        for key, value in calibration_report(loo, subset_correct).items()
                    },
                    "indist_aurc": aurc(indist[test_rows], subset_correct),
                    "indist_acc_at_0.7": 1.0
                    - risk_at_coverage(indist[test_rows], subset_correct, 0.7),
                    **{
                        f"indist_{key}": value
                        for key, value in calibration_report(
                            indist[test_rows], subset_correct
                        ).items()
                    },
                }
            )
    return rows


def shots_report(
    prepared: _Prepared,
    shots: list[int],
    *,
    seed: int = 0,
    C: float = RISK_C_DEFAULT,
    variant: str = PRODUCTION_VARIANT,
    baseline: str = "confidence",
    shot_seeds: Sequence[int] = (0,),
    include_others: bool = True,
) -> list[dict[str, Any]]:
    """Кривая «сколько строк нового вопроса нужно»: AURC на отложенном вопросе.

    Для каждого вопроса обучение идёт на валидации **остальных** вопросов плюс
    ``n`` строк валидации самого вопроса, оценка — на его тесте (тест не
    трогается). Точки вложены: одни и те же ``n`` строк берутся в одном порядке,
    поэтому рост кривой не объясняется случайной подменой примеров.

    ``include_others=False`` убирает чужие строки: обучение только на своих
    ``n`` строках. Это режим **первого** вопроса в сервисе, когда примеров других
    вопросов ещё нет (см. ``RiskAwareCascadeBackend._reference_rows``); точка
    ``n=0`` в этом режиме пропускается — обучать не на чем. Выборка строк при этом
    не меняется, поэтому кривые двух режимов попарно сопоставимы по
    ``(seed, shot_seed, вопрос, n)``.

    Для каждой точки считаются метрики и варианта ``variant``, и ``baseline`` —
    с ними потом работает hard gate (см. `gate_vs_baseline`,
    `GATE_SHOTS_METRICS`).

    ``shots=None`` — все свои строки (обучение на всех вопросах, верхняя
    граница), ``n = 0`` — перенос без своих данных (совпадает с ``--loo``).

    ``shot_seeds`` — несколько случайных **порядков** выбора своих строк (какие
    именно строки попали в первые ``n``). Строки отдаются по каждому порядку
    отдельно (ключ ``shot_seed``), усредняет их `average_shot_seeds` — так
    видно, насколько кривая зависит от удачного выбора примеров.
    """
    val_correct, test_correct = prepared.correct
    val_groups = np.asarray(prepared.groups[0])
    test_groups = np.asarray(prepared.groups[1])
    candidates = [name for name in (variant, baseline) if name in prepared.variants]

    rows: list[dict[str, Any]] = []
    for shot_seed in shot_seeds:
        for position, question in enumerate(prepared.questions):
            own = np.flatnonzero(val_groups == question)
            others = np.flatnonzero(val_groups != question)
            test_rows = test_groups == question
            subset_correct = test_correct[test_rows]
            # один порядок строк для всех точек кривой: точки вложены
            order = np.random.default_rng([shot_seed, position]).permutation(len(own))
            for n in [None, *shots]:
                picked = own if n is None else own[order[:n]]
                if n == 0 and not include_others:
                    continue  # без чужих строк n=0 — обучение на пустой выборке
                train_rows = (
                    picked if not include_others else np.concatenate([others, picked])
                )
                for name in candidates:
                    val_matrix, test_matrix = prepared.variants[name]
                    scores = _scores(
                        val_matrix[train_rows],
                        test_matrix[test_rows],
                        val_correct[train_rows],
                        seed=seed,
                        C=_c_for(prepared, name, C),
                    )
                    rows.append(
                        {
                            "variant": name,
                            "question": question,
                            "seed": seed,
                            "shot_seed": shot_seed,
                            "shots": n,
                            "n_own": len(picked),
                            "n_train": len(train_rows),
                            "n_test": int(test_rows.sum()),
                            **risk_report(scores, subset_correct),
                        }
                    )
    return rows


def average_shot_seeds(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Усреднить кривую по ``shot_seed``: ``(variant, seed, question, shots)``.

    Порядок выбора своих строк — nuisance-переменная: удачный порядок не должен
    выглядеть улучшением. Поэтому метрики усредняются по порядкам, а разброс
    хранится рядом (``<метрика>_shot_std``) и печатается отдельно. Если порядок
    один, ``std`` нулевой и строки эквивалентны исходным.
    """
    groups: dict[tuple[Any, Any, Any, Any], list[dict[str, Any]]] = {}
    for row in rows:
        key = (row["variant"], row.get("seed"), row["question"], row["shots"])
        groups.setdefault(key, []).append(row)

    averaged: list[dict[str, Any]] = []
    for group in groups.values():
        first = group[0]
        entry: dict[str, Any] = {
            "variant": first["variant"],
            "question": first["question"],
            "seed": first.get("seed"),
            "shots": first["shots"],
            "n_own": first["n_own"],
            "n_train": first["n_train"],
            "n_test": first["n_test"],
            "n_shot_seeds": len(group),
        }
        for column in SHOT_METRIC_COLUMNS:
            values = [float(row[column]) for row in group]
            entry[column] = float(np.mean(values))
            entry[f"{column}_shot_std"] = float(np.std(values))
        averaged.append(entry)
    return averaged


def gate_vs_baseline(
    rows: list[dict[str, Any]],
    metrics: tuple[GateMetric, ...],
    *,
    baseline: str = "confidence",
    tol: float | None = None,
) -> list[dict[str, Any]]:
    """Hard gate: каждый вариант против ``baseline`` по каждой (seed, вопрос).

    Строки соединяются по ``(seed, question, variant[, shots])``; ``question``
    равен ``None`` в A/B-строках (вопрос общий), ``shots`` — только у кривой
    few-shot. Возвращает вердикт по каждой паре: ``accepted`` ложно, если по
    **хотя бы одной** метрике вариант хуже baseline сильнее допуска — тогда от
    него отказываемся и остаёмся на baseline.
    """
    index = {
        (
            row.get("seed"),
            row.get("question"),
            row["variant"],
            row.get("shots"),
        ): row
        for row in rows
    }
    verdicts: list[dict[str, Any]] = []
    for (seed, question, variant, shots), row in index.items():
        if variant == baseline:
            continue
        base = index.get((seed, question, baseline, shots))
        if base is None:
            continue
        accepted, worse = gate_risk_vs_confidence(base, row, metrics, tol=tol)
        verdicts.append(
            {
                "variant": variant,
                "question": question,
                "seed": seed,
                "shots": shots,
                "accepted": accepted,
                "worse": worse,
            }
        )
    return verdicts


def aggregate_gate(verdicts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Доля замеров, где вариант прошёл гейт (лучший — 100%)."""
    by_point: dict[tuple[str, Any], list[dict[str, Any]]] = {}
    for verdict in verdicts:
        by_point.setdefault((verdict["variant"], verdict.get("shots")), []).append(
            verdict
        )

    aggregated = []
    for (variant, shots), group in sorted(
        by_point.items(), key=lambda item: str(item[0])
    ):
        accepted = sum(1 for verdict in group if verdict["accepted"])
        aggregated.append(
            {
                "variant": variant,
                "shots": shots,
                "n": len(group),
                "accepted": accepted,
                "accepted_rate": accepted / len(group),
                "failures": [
                    {
                        "question": verdict["question"],
                        "seed": verdict["seed"],
                        "worse": verdict["worse"],
                    }
                    for verdict in group
                    if not verdict["accepted"]
                ],
            }
        )
    return aggregated


def print_gate_report(
    aggregated: list[dict[str, Any]],
    *,
    context: str,
    metrics: tuple[GateMetric, ...] | None = None,
    tol: float | None = None,
) -> None:
    """Вердикты гейта: сколько замеров вариант прошёл против baseline."""
    if tol is not None:
        detail = f"tol={tol:g}"
    elif metrics is not None:
        tolerances = gate_tolerances(metrics)
        detail = "допуски " + ", ".join(
            f"{key} {value:g}" for key, value in tolerances.items()
        )
    else:
        detail = "допуск по метрике"
    print(
        f"\nhard gate, {context} (вариант принимается только если не хуже "
        f"baseline по всем метрикам, {detail}):"
    )
    print(f"{'variant':>24} {'своих строк':>12} {'пройдено':>9} {'доля':>6}")
    for entry in aggregated:
        label = "-" if entry["shots"] is None else str(entry["shots"])
        print(
            f"{entry['variant']:>24} {label:>12} "
            f"{entry['accepted']:>4}/{entry['n']:<4} {entry['accepted_rate']:>6.2f}"
        )
    for entry in aggregated:
        if entry["variant"] != PRODUCTION_VARIANT or not entry["failures"]:
            continue
        for failure in entry["failures"][:3]:
            print(
                f"  отказ {PRODUCTION_VARIANT} "
                f"(question={failure['question']}, seed={failure['seed']}): "
                + "; ".join(failure["worse"])
            )


COLUMNS = (
    "aurc",
    "eaurc",
    "coverage_at_5pct_risk",
    "acc_at_0.7_coverage",
    "ece",
    "brier",
)


def aggregate(
    rows_by_variant: dict[str, list[dict[str, float]]],
) -> list[dict[str, Any]]:
    """Среднее и разброс по seed'ам для каждой вариации."""
    aggregated = []
    for variant, rows in rows_by_variant.items():
        entry: dict[str, Any] = {"variant": variant, "n_seeds": len(rows)}
        for column in COLUMNS:
            values = [row[column] for row in rows]
            entry[column] = float(np.mean(values))
            entry[f"{column}_std"] = float(np.std(values))
        aggregated.append(entry)
    return aggregated


def print_report(aggregated: list[dict[str, Any]], n_seeds: int) -> None:
    print(f"\nmean ± std over {n_seeds} seeds (lower is better):")
    print(
        f"{'variant':>24} {'AURC':>16} {'eAURC':>16} {'acc@0.7':>16} "
        f"{'ECE':>16} {'Brier':>16}"
    )
    for entry in aggregated:
        cells = [
            f"{entry[c]:.4f}±{entry[f'{c}_std']:.4f}"
            for c in ("aurc", "eaurc", "acc_at_0.7_coverage", "ece", "brier")
        ]
        print(f"{entry['variant']:>24} " + " ".join(f"{c:>16}" for c in cells))


LOO_COLUMNS = (
    "loo_aurc",
    "loo_acc_at_0.7",
    "loo_ece",
    "loo_brier",
    "indist_aurc",
    "indist_acc_at_0.7",
    "indist_ece",
    "indist_brier",
)


def aggregate_loo(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Макро-среднее по вопросам, затем среднее и разброс по seed'ам."""
    per_seed: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in rows:
        per_seed.setdefault((row["variant"], row["seed"]), []).append(row)

    by_variant: dict[str, list[dict[str, float]]] = {}
    for (variant, _seed), group in per_seed.items():
        by_variant.setdefault(variant, []).append(
            {column: float(np.mean([row[column] for row in group]))
             for column in LOO_COLUMNS}
        )

    aggregated = []
    for variant, seeds in by_variant.items():
        entry: dict[str, Any] = {"variant": variant, "n_seeds": len(seeds)}
        for column in LOO_COLUMNS:
            values = [seed[column] for seed in seeds]
            entry[column] = float(np.mean(values))
            entry[f"{column}_std"] = float(np.std(values))
        aggregated.append(entry)
    return aggregated


def print_loo_report(aggregated: list[dict[str, Any]], n_seeds: int) -> None:
    """Сводка LOO: перенос против «знания вопроса»."""
    print(
        f"\nleave-one-question-out, макро-среднее по вопросам ± std over "
        f"{n_seeds} seeds:"
    )
    print(
        f"{'variant':>24} {'LOO AURC':>16} {'in-dist AURC':>13} "
        f"{'разрыв':>9} {'LOO acc@0.7':>12}"
    )
    for entry in aggregated:
        print(
            f"{entry['variant']:>24} "
            f"{entry['loo_aurc']:.4f}±{entry['loo_aurc_std']:.4f} "
            f"{entry['indist_aurc']:>13.4f} "
            f"{entry['indist_aurc'] - entry['loo_aurc']:>9.4f} "
            f"{entry['loo_acc_at_0.7']:>12.4f}"
        )


def print_loo_calibration(aggregated: list[dict[str, Any]], n_seeds: int) -> None:
    """Калибровка ``P(ошибка)``: LOO против in-dist (меньше лучше)."""
    print(f"\nкалибровка P(ошибка), макро по вопросам ± std over {n_seeds} seeds:")
    print(
        f"{'variant':>24} {'LOO ECE':>16} {'in-dist ECE':>13} "
        f"{'LOO Brier':>16} {'in-dist Brier':>14}"
    )
    for entry in aggregated:
        print(
            f"{entry['variant']:>24} "
            f"{entry['loo_ece']:.4f}±{entry['loo_ece_std']:.4f} "
            f"{entry['indist_ece']:>13.4f} "
            f"{entry['loo_brier']:.4f}±{entry['loo_brier_std']:.4f} "
            f"{entry['indist_brier']:>14.4f}"
        )


SHOT_METRIC_COLUMNS = (
    "aurc",
    "eaurc",
    "coverage_at_5pct_risk",
    "acc_at_0.7_coverage",
    "ece",
    "brier",
)
SHOT_COLUMNS = (*SHOT_METRIC_COLUMNS, "n_own")


def aggregate_shots(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Макро-среднее по вопросам, затем среднее и разброс по seed'ам.

    Строки должны быть уже усреднены по ``shot_seed`` (`average_shot_seeds`):
    разброс по порядкам примеров едет рядом как ``<метрика>_shot_std`` и тоже
    усредняется (сколько в среднем «гуляет» метрика от порядка).
    """
    per_seed: dict[tuple[Any, Any, int], list[dict[str, Any]]] = {}
    for row in rows:
        key = (row["variant"], row["shots"], row["seed"])
        per_seed.setdefault(key, []).append(row)

    by_point: dict[tuple[Any, Any], list[dict[str, float]]] = {}
    for (variant, shots, _seed), group in per_seed.items():
        by_point.setdefault((variant, shots), []).append(
            {
                column: float(np.mean([row[column] for row in group]))
                for column in SHOT_COLUMNS
            }
            | {
                f"{column}_shot_std": float(
                    np.mean([row.get(f"{column}_shot_std", 0.0) for row in group])
                )
                for column in SHOT_METRIC_COLUMNS
            }
        )

    aggregated = []
    for (variant, shots), entries in by_point.items():
        entry: dict[str, Any] = {
            "variant": variant,
            "shots": shots,
            "n_seeds": len(entries),
        }
        for column in SHOT_COLUMNS:
            values = [item[column] for item in entries]
            entry[column] = float(np.mean(values))
            entry[f"{column}_std"] = float(np.std(values))
        for column in SHOT_METRIC_COLUMNS:
            entry[f"{column}_shot_std"] = float(
                np.mean([item[f"{column}_shot_std"] for item in entries])
            )
        aggregated.append(entry)
    return aggregated


def _shots_order(entry: dict[str, Any]) -> tuple[str, int, float]:
    """По варианту, затем числовые точки по возрастанию, «все» — последней."""
    variant = str(entry["variant"])
    if entry["shots"] is None:
        return (variant, 1, 0.0)
    return (variant, 0, float(entry["shots"]))


def print_shots_report(aggregated: list[dict[str, Any]], n_seeds: int) -> None:
    """Кривая: AURC на новом вопросе против числа его размеченных строк."""
    print(
        f"\nсколько строк нового вопроса нужно (макро по вопросам "
        f"± std over {n_seeds} seeds):"
    )
    print(
        f"{'variant':>24} {'своих строк':>12} {'AURC':>16} {'acc@0.7':>8} "
        f"{'ECE':>8} {'Brier':>8} {'взято':>6}"
    )
    for entry in sorted(aggregated, key=_shots_order):
        label = "все (in-dist)" if entry["shots"] is None else str(entry["shots"])
        print(
            f"{entry['variant']:>24} {label:>12} "
            f"{entry['aurc']:.4f}±{entry['aurc_std']:.4f} "
            f"{entry['acc_at_0.7_coverage']:>8.4f} "
            f"{entry['ece']:>8.4f} {entry['brier']:>8.4f} "
            f"{entry['n_own']:>6.1f}"
        )


def print_shots_stability(
    aggregated: list[dict[str, Any]], shot_seeds: Sequence[int]
) -> None:
    """Насколько кривая зависит от порядка выбора своих строк (std по shot_seed)."""
    print(
        f"\nразброс по порядку примеров (std по shot_seed={list(shot_seeds)}):"
    )
    print(f"{'variant':>24} {'своих строк':>12} {'σ(AURC)':>9} {'σ(ECE)':>9}")
    for entry in sorted(aggregated, key=_shots_order):
        label = "все (in-dist)" if entry["shots"] is None else str(entry["shots"])
        print(
            f"{entry['variant']:>24} {label:>12} "
            f"{entry['aurc_shot_std']:>9.4f} {entry['ece_shot_std']:>9.4f}"
        )


def print_stability(report: dict[str, dict[str, float]], *, context: str) -> None:
    """CI среднего различия candidate−baseline («+» = кандидат хуже).

    Даёт «эффект или шум» по-настоящему: интервал строится перевыборкой пар
    «seed×вопрос», а не сравнением с постоянным допуском.
    """
    print(f"\nстабильность, {context} (разница production−confidence, «+» — хуже):")
    print(
        f"{'метрика':>26} {'mean':>9} {'95% CI':>20} {'доля хуже':>10} {'n':>4}"
    )
    for key, entry in report.items():
        if not np.isfinite(entry["mean"]):
            continue
        flag = "  ← эффект" if entry["significant"] else ""
        print(
            f"{key:>26} {entry['mean']:>9.4f} "
            f"[{entry['lo']:>7.4f}, {entry['hi']:>7.4f}] "
            f"{entry['share_worse']:>10.2f} {entry['n']:>4}{flag}"
        )


def worst_case_by_question(
    rows: list[dict[str, Any]],
    metric: str,
    *,
    variant: str = PRODUCTION_VARIANT,
) -> dict[str, Any] | None:
    """Худший вопрос по метрике (для «меньше лучше» — максимум).

    Макро-среднее прячет провал на одном вопросе; worst-case — минимальная
    страховка от этого (bootstrap-CI в `stability_report` работает по парам).
    """
    selected = [
        row for row in rows
        if row["variant"] == variant and metric in row and row[metric] is not None
    ]
    if not selected:
        return None
    worst = max(selected, key=lambda row: float(row[metric]))
    return {
        "question": worst.get("question"),
        "seed": worst.get("seed"),
        "metric": metric,
        "value": float(worst[metric]),
    }


def print_loo_per_question(
    rows: list[dict[str, Any]],
    variant: str = PRODUCTION_VARIANT,
    seed: int = 0,
) -> None:
    """По вопросам (один seed): видно, какой вопрос «переносится», а какой нет."""
    selected = [
        row for row in rows
        if row["variant"] == variant and row["seed"] == seed
    ]
    if not selected:
        return
    print(f"\nper-question LOO (variant={variant}, seed={seed}):")
    print(f"{'question':>16} {'n_val':>7} {'n_test':>7} {'LOO AURC':>10} "
          f"{'in-dist':>9}")
    for row in selected:
        print(
            f"{row['question']:>16} {row['n_val']:>7} {row['n_test']:>7} "
            f"{row['loo_aurc']:>10.4f} {row['indist_aurc']:>9.4f}"
        )


def main(argv: list[str] | None = None) -> list[dict[str, Any]]:
    parser = argparse.ArgumentParser(description="A/B risk feature set")
    parser.add_argument("--specs", choices=["support", "public", "routerbench"],
                        default="routerbench")
    parser.add_argument("--encoder", default="hash")
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--train-steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument(
        "--C", type=float, default=RISK_C_DEFAULT,
        help=f"сила L2 (по умолчанию {RISK_C_DEFAULT}, дефолт продукта)",
    )
    parser.add_argument("--C-grid", type=float, nargs="+", default=None,
                        help="сетка сил L2 для вариантов production[C]: "
                             "признаки кодируются один раз, сетка по C не "
                             "требует повторного кодирования")
    parser.add_argument("--scale", action="store_true",
                        help="стандартизировать признаки по валидации "
                             "(проверка чувствительности к масштабу)")
    parser.add_argument("--loo", action="store_true",
                        help="leave-one-question-out: обучать риск-модель без "
                             "одного вопроса и замерять на нём (несмещённая "
                             "оценка переноса на новый вопрос)")
    parser.add_argument("--state", action="store_true",
                        help="добавить варианты с состоянием энкодера: "
                             "state[C] и prod+state[C]")
    parser.add_argument("--state-C", type=float, nargs="+", default=None,
                        help="сила L2 для вариантов с состоянием (по умолчанию "
                             "0.05; можно задать несколько значений)")
    parser.add_argument("--question", action="store_true",
                        help="добавить варианты с явным свободным членом на "
                             "вопрос: prod+question и no_n_classes+question")
    parser.add_argument("--ablate", action="store_true",
                        help="добавить варианты production минус один новый "
                             "признак: no_n_classes, no_log_odds_gap, no_noul_logit")
    parser.add_argument("--qmean", action="store_true",
                        help="добавить варианты со средним пулированным вектором "
                             "вопроса: qmean[C] и prod+qmean[C]")
    parser.add_argument("--qmean-C", type=float, nargs="+", default=None,
                        help="сила L2 для qmean-вариантов (по умолчанию 0.05)")
    parser.add_argument("--shots", type=int, nargs="+", default=None,
                        help="кривая дообучения на новом вопросе: сколько строк "
                             "валидации вопроса добавить в обучение (0 — перенос "
                             "без своих данных; 0 включается всегда)")
    parser.add_argument("--shot-seeds", type=int, nargs="+", default=[0],
                        help="случайные порядки выбора своих строк: усредняем по "
                             "ним, чтобы кривая не зависела от удачного выбора")
    parser.add_argument("--shots-no-others", action="store_true",
                        help="кривая без чужих строк: обучать риск-модель только "
                             "на своих n строках вопроса — режим первого вопроса "
                             "в сервисе, когда примеров других вопросов ещё нет "
                             "(точка n=0 в этом режиме не считается)")
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    shots = None if not args.shots else sorted({0, *args.shots})
    specs = _specs(args.specs)
    rows_by_variant: dict[str, list[dict[str, float]]] = {}
    loo_rows: list[dict[str, Any]] = []
    shots_rows: list[dict[str, Any]] = []
    for seed in args.seeds:
        splits = _load_splits(SimpleNamespace(
            specs=args.specs, dataset=None, limit=args.limit, seed=seed,
            shot="0shot",
        ))
        train, test = splits["train"], splits["test"]
        val = splits.get("val") or train
        encoder = _encoder(args.encoder, args.dim, args.device, args.dtype)
        heads, _ = train_benchmark(
            specs, train, encoder, train_steps=args.train_steps, lr=args.lr,
            seed=seed,
        )
        prepared = prepare(
            heads, encoder, specs, val, test, scale=args.scale,
            with_state=args.state, state_cs=args.state_C,
            with_question=args.question, ablate=args.ablate,
            with_qmean=args.qmean, qmean_cs=args.qmean_C,
            c_grid=args.C_grid,
        )
        for row in run_ab(prepared, seed=seed, C=args.C):
            rows_by_variant.setdefault(row["variant"], []).append(row)
        if args.loo:
            loo_rows.extend(run_loo(prepared, seed=seed, C=args.C))
        if shots is not None:
            shots_rows.extend(
                shots_report(
                    prepared, shots, seed=seed, C=args.C,
                    shot_seeds=args.shot_seeds,
                    include_others=not args.shots_no_others,
                )
            )
        if seed == args.seeds[0]:
            print(f"specs={args.specs} encoder={args.encoder} "
                  f"train={len(train)} val={len(val)} test={len(test)}")

    aggregated = aggregate(rows_by_variant)
    print_report(aggregated, len(args.seeds))
    ab_rows = [row for rows in rows_by_variant.values() for row in rows]
    gate_ab = gate_vs_baseline(ab_rows, GATE_AB_METRICS)
    print_gate_report(
        aggregate_gate(gate_ab), context="in-dist (та же база)",
        metrics=GATE_AB_METRICS,
    )
    loo_aggregated = aggregate_loo(loo_rows) if args.loo else []
    gate_loo: list[dict[str, Any]] = []
    if args.loo:
        print_loo_report(loo_aggregated, len(args.seeds))
        print_loo_calibration(loo_aggregated, len(args.seeds))
        print_loo_per_question(loo_rows)
        gate_loo = gate_vs_baseline(loo_rows, GATE_LOO_METRICS)
        print_gate_report(
            aggregate_gate(gate_loo), context="на новом вопросе (LOO)",
            metrics=GATE_LOO_METRICS,
        )
    gate_shots: list[dict[str, Any]] = []
    shots_averaged: list[dict[str, Any]] = []
    if shots is not None:
        # порядок выбора своих строк — nuisance: усредняем до гейта
        shots_averaged = average_shot_seeds(shots_rows)
        shots_aggregated = aggregate_shots(shots_averaged)
        print_shots_report(shots_aggregated, len(args.seeds))
        if len(args.shot_seeds) > 1:
            print_shots_stability(shots_aggregated, args.shot_seeds)
        gate_shots = gate_vs_baseline(shots_averaged, GATE_SHOTS_METRICS)
        print_gate_report(
            aggregate_gate(gate_shots), context="кривая few-shot (новый вопрос)",
            metrics=GATE_SHOTS_METRICS,
        )

    # стабильность: CI среднего различия по парам (seed, вопрос) — «эффект или шум»
    stability: dict[str, Any] = {
        "in_dist": stability_report(
            paired_differences(ab_rows, GATE_AB_METRICS)
        ),
    }
    print_stability(stability["in_dist"], context="in-dist (пул)")
    if args.loo:
        stability["loo"] = stability_report(
            paired_differences(loo_rows, GATE_LOO_METRICS)
        )
        stability["loo_worst_auc"] = worst_case_by_question(loo_rows, "loo_aurc")
        print_stability(stability["loo"], context="новый вопрос (LOO)")
        worst = stability["loo_worst_auc"]
        if worst is not None:
            print(
                f"худший вопрос по LOO AURC: {worst['question']} "
                f"(seed {worst['seed']}) = {worst['value']:.4f}"
            )
    if shots is not None:
        stability["shots"] = {}
        for point in sorted({row["shots"] for row in shots_averaged},
                            key=lambda value: (value is None, value)):
            rows_at_point = [
                row for row in shots_averaged if row["shots"] == point
            ]
            stability["shots"][str(point)] = stability_report(
                paired_differences(rows_at_point, GATE_SHOTS_METRICS)
            )
        stability["shots_worst_auc"] = worst_case_by_question(
            shots_averaged, "aurc"
        )

    if args.out_dir is not None:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "specs": args.specs,
            "encoder": args.encoder,
            "seeds": args.seeds,
            "shot_seeds": args.shot_seeds,
            "scaled": args.scale,
            "feature_names": list(FEATURE_NAMES),
            "extra_features": list(EXTRA_FEATURE_NAMES),
            "state_cs": args.state_C if args.state else None,
            "qmean_cs": args.qmean_C if args.qmean else None,
            "c_grid": args.C_grid or None,
            "ablated": args.ablate,
            "per_seed": rows_by_variant,
            "aggregate": aggregated,
            "loo": aggregate_loo(loo_rows) if args.loo else None,
            "loo_per_question": loo_rows,
            "gate": {
                "tol": GATE_TOL,
                "tolerances": gate_tolerances(GATE_AB_METRICS)
                | gate_tolerances(GATE_LOO_METRICS)
                | gate_tolerances(GATE_SHOTS_METRICS),
                "baseline": "confidence",
                "in_dist": aggregate_gate(gate_ab),
                "loo": aggregate_gate(gate_loo) if args.loo else None,
                "shots": aggregate_gate(gate_shots) if shots is not None else None,
                "verdicts_in_dist": gate_ab,
                "verdicts_loo": gate_loo,
                "verdicts_shots": gate_shots,
            },
            "shots": (
                aggregate_shots(shots_averaged) if shots is not None else None
            ),
            "shots_per_question": shots_averaged or None,
            # сырые точки по каждому порядку примеров — для оценки шума гейта
            "shots_per_shot_seed": shots_rows or None,
            "stability": stability,
        }
        (args.out_dir / "features.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"metrics -> {args.out_dir / 'features.json'}")
    return aggregated


if __name__ == "__main__":
    main()
