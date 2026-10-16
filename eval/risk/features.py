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
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from eval.benchmark.core import TextEncoder, encode_examples, train_benchmark
from eval.calibration.selective import aurc, coverage_at_risk, eaurc, risk_at_coverage
from eval.report.run import _encoder, _load_splits, _specs
from eval.risk.core import decisions_for_examples, flatten
from src.decision_head.heads import OptionScoringChoiceHead
from src.risk import FEATURE_NAMES, features_matrix
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
    features: np.ndarray, correct: np.ndarray, *, seed: int = 0, C: float = 1.0
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


def risk_report(scores: np.ndarray, correct: np.ndarray) -> dict[str, float]:
    """AURC/eAURC, покрытие при бюджете 5% и точность при покрытии 0.7."""
    return {
        "aurc": aurc(scores, correct),
        "eaurc": eaurc(scores, correct),
        "coverage_at_5pct_risk": coverage_at_risk(scores, correct, 0.05),
        "acc_at_0.7_coverage": 1.0 - risk_at_coverage(scores, correct, 0.7),
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
) -> _Prepared:
    """Один раз считает решения, признаки и корректность для A/B и LOO.

    ``with_state`` добавляет варианты с состоянием энкодера (см.
    `with_state_variants`), ``state_cs`` — их силы L2, ``with_question`` —
    варианты с явным свободным членом на вопрос (см. `question_onehot`).
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
        scale=scale, question=with_question,
    )
    cs: dict[str, float] = {}
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
    prepared: _Prepared, *, seed: int = 0, C: float = 1.0
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
    prepared: _Prepared, *, seed: int = 0, C: float = 1.0
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
                    "loo_acc_at_0.7": 1.0
                    - risk_at_coverage(loo, subset_correct, 0.7),
                    "indist_aurc": aurc(indist[test_rows], subset_correct),
                    "indist_acc_at_0.7": 1.0
                    - risk_at_coverage(indist[test_rows], subset_correct, 0.7),
                }
            )
    return rows


COLUMNS = ("aurc", "eaurc", "coverage_at_5pct_risk", "acc_at_0.7_coverage")


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
    print(f"\nmean ± std over {n_seeds} seeds (lower AURC/eAURC is better):")
    print(f"{'variant':>24} {'AURC':>16} {'eAURC':>16} {'acc@0.7':>16}")
    for entry in aggregated:
        cells = [
            f"{entry[c]:.4f}±{entry[f'{c}_std']:.4f}"
            for c in ("aurc", "eaurc", "acc_at_0.7_coverage")
        ]
        print(f"{entry['variant']:>24} " + " ".join(f"{c:>16}" for c in cells))


LOO_COLUMNS = ("loo_aurc", "loo_acc_at_0.7", "indist_aurc", "indist_acc_at_0.7")


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
    parser.add_argument("--C", type=float, default=1.0)
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
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    specs = _specs(args.specs)
    rows_by_variant: dict[str, list[dict[str, float]]] = {}
    loo_rows: list[dict[str, Any]] = []
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
            with_question=args.question,
        )
        for row in run_ab(prepared, seed=seed, C=args.C):
            rows_by_variant.setdefault(row["variant"], []).append(row)
        if args.loo:
            loo_rows.extend(run_loo(prepared, seed=seed, C=args.C))
        if seed == args.seeds[0]:
            print(f"specs={args.specs} encoder={args.encoder} "
                  f"train={len(train)} val={len(val)} test={len(test)}")

    aggregated = aggregate(rows_by_variant)
    print_report(aggregated, len(args.seeds))
    loo_aggregated = aggregate_loo(loo_rows) if args.loo else []
    if args.loo:
        print_loo_report(loo_aggregated, len(args.seeds))
        print_loo_per_question(loo_rows)

    if args.out_dir is not None:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "specs": args.specs,
            "encoder": args.encoder,
            "seeds": args.seeds,
            "scaled": args.scale,
            "feature_names": list(FEATURE_NAMES),
            "extra_features": list(EXTRA_FEATURE_NAMES),
            "state_cs": args.state_C if args.state else None,
            "per_seed": rows_by_variant,
            "aggregate": aggregated,
            "loo": aggregate_loo(loo_rows) if args.loo else None,
            "loo_per_question": loo_rows,
        }
        (args.out_dir / "features.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"metrics -> {args.out_dir / 'features.json'}")
    return aggregated


if __name__ == "__main__":
    main()
