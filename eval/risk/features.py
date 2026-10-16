"""A/B-замер состава признаков оценщика риска.

Признаки решения считает `src.risk.decision_features` — это то, что реально
подключается в сервисе. Этот CLI отвечает на вопрос: **какие** из них дают
сигнал об ошибке, добавляясь к трём базовым (`confidence, entropy, margin`).

Проверяются:

- `base(3)` — три базовых признака: прежняя модель (до расширения);
- `+log_odds_gap`, `+n_classes`, `+noul_logit` — base плюс **один** новый
  признак (видно, кто тянет выигрыш);
- `production` — все признаки из `FEATURE_NAMES`, то есть ровно то, что
  обучает `training.recipe.risk` и что проверяется в `eval.risk.run`
  (числа должны совпадать с `risk_model` оттуда);
- `+option_similarity` — production плюс признак, которому нужны
  **внутренности головы** (средняя попарная косинусная близость направлений
  вариантов) и потому поле в схеме `DecisionResult`.

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
    *,
    scale: bool = False,
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
    return variants


def _flat(decisions: dict[str, list[DecisionResult]]) -> list[DecisionResult]:
    return [decision for group in decisions.values() for decision in group]


def run_ab(
    heads,
    encoder: TextEncoder,
    specs: dict[str, QuestionSpec],
    val: list[dict[str, Any]],
    test: list[dict[str, Any]],
    *,
    seed: int = 0,
    C: float = 1.0,
    scale: bool = False,
) -> list[dict[str, float]]:
    """Подгоняет риск-модели на ``val`` и считает метрики на ``test``."""
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
        scale=scale,
    )

    rows = []
    for name, (train_matrix, eval_matrix) in variants.items():
        weights, bias = _fit_logistic(train_matrix, val_correct, seed=seed, C=C)
        # модель предсказывает P(ошибка); ранжировщику нужен балл «чем больше,
        # тем лучше» — как в ``eval.risk.run``
        error = 1.0 / (1.0 + np.exp(-(eval_matrix @ weights + bias)))
        rows.append({"variant": name, **risk_report(1.0 - error, test_correct)})
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
    print(f"{'variant':>20} {'AURC':>16} {'eAURC':>16} {'acc@0.7':>16}")
    for entry in aggregated:
        cells = [
            f"{entry[c]:.4f}±{entry[f'{c}_std']:.4f}"
            for c in ("aurc", "eaurc", "acc_at_0.7_coverage")
        ]
        print(f"{entry['variant']:>20} " + " ".join(f"{c:>16}" for c in cells))


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
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    specs = _specs(args.specs)
    rows_by_variant: dict[str, list[dict[str, float]]] = {}
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
        for row in run_ab(heads, encoder, specs, val, test, seed=seed, C=args.C,
                          scale=args.scale):
            rows_by_variant.setdefault(row["variant"], []).append(row)
        if seed == args.seeds[0]:
            print(f"specs={args.specs} encoder={args.encoder} "
                  f"train={len(train)} val={len(val)} test={len(test)}")

    aggregated = aggregate(rows_by_variant)
    print_report(aggregated, len(args.seeds))
    if args.out_dir is not None:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "specs": args.specs,
            "encoder": args.encoder,
            "seeds": args.seeds,
            "scaled": args.scale,
            "feature_names": list(FEATURE_NAMES),
            "extra_features": list(EXTRA_FEATURE_NAMES),
            "per_seed": rows_by_variant,
            "aggregate": aggregated,
        }
        (args.out_dir / "features.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"metrics -> {args.out_dir / 'features.json'}")
    return aggregated


if __name__ == "__main__":
    main()
