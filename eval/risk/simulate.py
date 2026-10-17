"""Синтетические границы применимости риск-оценщика (Шаг 4 roadmap).

Реальные прогоны (`eval.risk.features`, Шаг 3) дали три результата, которые на
живых данных трудно развести:

1. признак-суррогат `n_classes` даёт почти весь выигрыш **в пуле** вопросов и не
   стоит ничего **в переносе**;
2. явный свободный член на вопрос (`one-hot`) работает как суррогат, но честнее
   (не «узнаёт» вопрос по числу исходов);
3. `noul_logit` — наоборот: помогает в пуле и значимо ломает перенос на обоих
   энкодерах.

Симуляция изолирует **структуру**: что может и что не может дать «узнавание
вопроса» по признакам, и какой именно признак может испортить перенос.

Модель данных. У каждого решения есть латентная сложность ``z ~ U(0, 1)``, у
вопроса — сдвиг ``intercepts[q]`` (вероятность ошибки при ``z = 0.5``) и сила
внутривопросного сигнала ``slopes[q]``:

    p_err = clip(intercepts[q] + slopes[q] * (z - 0.5), eps, 1 - eps)
    correct ~ Bernoulli(1 - p_err)

Наблюдаемые признаки — зашумлённые виды ``z``:

- ``confidence, entropy, margin`` — базовые (первые три);
- ``log_odds_gap`` — честный внутривопросный сигнал, одинаковый на всех вопросах;
- ``n_classes`` — атрибут **вопроса** (суррогат номера вопроса);
- ``noul_logit`` — внутривопросный признак, у которого связь с ошибкой
  **дрейфует**: на обучающих вопросах он как `confidence`, а на отложенном —
  перевёрнут (``дрейф`` в сценарии). Это модель «спайкового» признака, который
  подгоняется под обучающие вопросы и перестаёт работать на новом.

Отсюда вся механика:

- внутри одного вопроса ``n_classes`` **константа**, поэтому в балле
  ранжирования она сдвигает все строки одинаково и на AURC одного вопроса
  повлиять не может — на этом стоит вывод «перенос `n_classes` бесплатен»;
- в пуле вопросов та же константа выравнивает межвопросный порядок — на этом
  стоит выигрыш in-dist;
- перенос ломает только признак, у которого **дрейфует сама связь** с ошибкой
  (`noul_logit`), потому что его наклон, подобранный на обучающих вопросах, на
  новом неверен.

Сравниваются варианты (как в `eval.risk.features`):

- ``confidence`` — сырой балл (`eval.risk.run`); нижняя граница;
- ``base(3)`` — три базовых признака;
- ``+n_classes`` — базовые плюс суррогат номера вопроса (изолирует суррогат);
- ``production`` — всё, что подключено в сервисе (`+log_odds_gap +noul_logit`);
- ``no_n_classes+question`` — `production` без суррогата, но с one-hot вопроса
  (в сервисе не подключено, проверяется как честная замена).

Сценарии:

- ``S1`` коллизия классов: ``n_classes`` у всех вопросов один, сдвиги разные.
  Суррогат не различает вопросы → выигрыш in-dist исчезает.
- ``S2`` без коллизии: ``n_classes`` различает вопросы. Выигрыш in-dist есть,
  а в переносе суррогат инертен (константа внутри вопроса).
- ``S3`` дрейф: ``noul_logit`` перевёрнут на отложенном вопросе. Структурный
  суррогат бессилен — перенос портится именно из-за дрейфующего признака; плюс
  кривая few-shot, чтобы видеть, снимают ли свои строки эту потерю.
- ``S4`` кривая few-shot для ``S1``: сколько своих строк возвращают порядок.
- ``S5`` калибровка при сдвиге: ECE/Brier для ``S1`` и ``S4`` (отдельный отчёт).
- ``S6`` худший случай: отложенный вопрос вне диапазона обучения и по сдвигу,
  и по наклону, плюс дрейф.

Запуск:

    python -m eval.risk.simulate --seeds 0 1 2
    python -m eval.risk.simulate --scenarios S1 S2 S3 S5 --out-dir data/risk_simulate
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from eval.risk.features import _fit_logistic, risk_report

EPS = 1e-6
#: Имена столбцов синтетических признаков (порядок = порядок в матрице).
FEATURE_COLUMNS = (
    "confidence",
    "entropy",
    "margin",
    "log_odds_gap",
    "noul_logit",
)
#: Число «базовых» признаков (как `BASE_WIDTH` в замерной части).
BASE_WIDTH = 3
#: Варианты набора признаков — те же имена, что в замерной части.
VARIANTS = (
    "confidence",
    "base(3)",
    "+n_classes",
    "production",
    "no_n_classes+question",
)
#: Вложенные точки кривой few-shot.
SHOT_POINTS = (0, 1, 2, 5, 10, 20)
#: Метрики, которые агрегируются и печатаются.
METRIC_KEYS = (
    "aurc",
    "eaurc",
    "coverage_at_5pct_risk",
    "acc_at_0.7_coverage",
    "ece",
    "brier",
)


@dataclass(frozen=True)
class Scenario:
    """Синтетический набор вопросов.

    ``n_classes[q]`` — суррогат номера вопроса, ``intercepts[q]`` — вероятность
    ошибки вопроса при ``z = 0.5``, ``slopes[q]`` — сила внутривопросного
    сигнала, ``held_out`` — вопрос, убранный из обучения в LOO-проверке,
    ``drift`` — перевёрнут ли ``noul_logit`` на отложенном вопросе.
    """

    name: str
    n_classes: tuple[int, ...]
    intercepts: tuple[float, ...]
    slopes: tuple[float, ...]
    held_out: int
    drift: bool = False
    note: str = ""

    @property
    def n_questions(self) -> int:
        return len(self.n_classes)


SCENARIOS: dict[str, Scenario] = {
    "S1": Scenario(
        name="S1",
        n_classes=(3, 3, 3),
        intercepts=(0.10, 0.30, 0.50),
        slopes=(0.30, 0.30, 0.30),
        held_out=2,
        note="коллизия классов: суррогат не различает вопросы",
    ),
    "S2": Scenario(
        name="S2",
        n_classes=(2, 3, 4),
        intercepts=(0.10, 0.30, 0.50),
        slopes=(0.30, 0.30, 0.30),
        held_out=2,
        note="без коллизии: суррогат различает вопросы",
    ),
    "S3": Scenario(
        name="S3",
        n_classes=(2, 3, 4),
        intercepts=(0.10, 0.30, 0.50),
        slopes=(0.30, 0.30, 0.30),
        held_out=2,
        drift=True,
        note="дрейф признака: noul_logit перевёрнут на новом вопросе",
    ),
    "S4": Scenario(
        name="S4",
        n_classes=(3, 3, 3),
        intercepts=(0.10, 0.30, 0.50),
        slopes=(0.30, 0.30, 0.30),
        held_out=2,
        note="коллизия классов + кривая few-shot",
    ),
    "S6": Scenario(
        name="S6",
        n_classes=(2, 3, 4, 5),
        intercepts=(0.10, 0.20, 0.45, 0.50),
        slopes=(0.60, 0.45, 0.25, 0.10),
        held_out=0,
        drift=True,
        note="худший случай: вопрос вне диапазона обучения, плюс дрейф",
    ),
}
#: Сценарии, для которых считается кривая few-shot.
WITH_SHOTS = frozenset({"S3", "S4"})


def _onehot(question: np.ndarray, dim: int) -> np.ndarray:
    """One-hot по вопросам; у отсутствующего в обучении столбца вес будет 0."""
    out = np.zeros((len(question), dim))
    out[np.arange(len(question)), question.astype(int)] = 1.0
    return out


@dataclass(frozen=True)
class _Data:
    """Строки набора в порядке question-major (как в замерной части)."""

    matrix: np.ndarray  # [n, len(FEATURE_COLUMNS)]
    n_classes: np.ndarray  # [n]
    question: np.ndarray  # [n]
    correct: np.ndarray  # [n]

    def subset(self, mask: np.ndarray) -> _Data:
        return _Data(
            self.matrix[mask],
            self.n_classes[mask],
            self.question[mask],
            self.correct[mask],
        )


def _concat(*parts: _Data) -> _Data:
    return _Data(
        np.vstack([p.matrix for p in parts]),
        np.concatenate([p.n_classes for p in parts]),
        np.concatenate([p.question for p in parts]),
        np.concatenate([p.correct for p in parts]),
    )


def _rows_for_question(
    spec: Scenario, question: int, n: int, rng: np.random.Generator, noise: float
) -> tuple[np.ndarray, np.ndarray]:
    """Латентная сложность → признаки и метки для одного вопроса."""
    z = rng.uniform(0.0, 1.0, size=n)
    p_err = np.clip(
        spec.intercepts[question] + spec.slopes[question] * (z - 0.5), EPS, 1.0 - EPS
    )
    correct = (rng.uniform(size=n) > p_err).astype(int)

    def noisy(signal: np.ndarray) -> np.ndarray:
        return np.clip(signal + noise * rng.normal(size=n), 0.0, 1.0)

    easy = 1.0 - z  # «чем больше, тем легче» — сигнал «правильно»
    drift = spec.drift and question == spec.held_out
    matrix = np.column_stack(
        [
            noisy(easy),  # confidence
            noisy(z),  # entropy (растёт с трудностью)
            noisy(easy),  # margin
            noisy(easy),  # log_odds_gap — честный внутривопросный сигнал
            noisy(z if drift else easy),  # noul_logit — знак дрейфует
        ]
    )
    return matrix, correct


def _dataset(
    spec: Scenario, *, n: int, rng: np.random.Generator, noise: float
) -> _Data:
    matrices, n_classes, questions, correct = [], [], [], []
    for question in range(spec.n_questions):
        matrix, labels = _rows_for_question(spec, question, n, rng, noise)
        matrices.append(matrix)
        n_classes.append(np.full(n, spec.n_classes[question], dtype=float))
        questions.append(np.full(n, question, dtype=float))
        correct.append(labels)
    return _Data(
        np.vstack(matrices),
        np.concatenate(n_classes),
        np.concatenate(questions),
        np.concatenate(correct),
    )


def _features(data: _Data, name: str, *, dim: int) -> np.ndarray:
    """Матрица признаков варианта."""
    base = data.matrix[:, :BASE_WIDTH]
    extras = data.matrix[:, BASE_WIDTH:]
    if name == "confidence":
        return data.matrix[:, [0]]
    if name == "base(3)":
        return base
    if name == "+n_classes":
        return np.column_stack([base, data.n_classes])
    if name == "production":
        return np.column_stack([base, extras, data.n_classes])
    if name == "no_n_classes+question":
        return np.column_stack([base, extras, _onehot(data.question, dim)])
    raise ValueError(f"неизвестный вариант: {name}")


def _scores(
    name: str,
    train: _Data,
    test: _Data,
    *,
    dim: int,
    seed: int,
    C: float,
) -> np.ndarray:
    """Балл «решение верно» на ``test`` (больше — лучше)."""
    if name == "confidence":
        # Сырой балл, как baseline в `eval.risk.run`. В `eval.risk.features`
        # baseline — обученная логистика на одном признаке; из-за переворота
        # знака при слабом сигнале она не эквивалентна сырому баллу, поэтому
        # здесь берём честный сырой вариант.
        return test.matrix[:, 0]
    weights, bias = _fit_logistic(
        _features(train, name, dim=dim), train.correct, seed=seed, C=C
    )
    error = 1.0 / (1.0 + np.exp(-(_features(test, name, dim=dim) @ weights + bias)))
    return 1.0 - error


def _mean_rows(
    rows: list[dict[str, Any]], group_keys: tuple[str, ...]
) -> list[dict[str, Any]]:
    """Среднее и std по seed'ам для каждой группы."""
    groups: dict[tuple, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(tuple(row[key] for key in group_keys), []).append(row)
    aggregated = []
    for key, group in groups.items():
        entry: dict[str, Any] = dict(zip(group_keys, key))
        for metric in METRIC_KEYS:
            values = [row[metric] for row in group]
            entry[metric] = float(np.mean(values))
            entry[f"{metric}_std"] = float(np.std(values))
        entry["n"] = len(group)
        aggregated.append(entry)
    return aggregated


def scenario_report(
    spec: Scenario,
    *,
    seeds: list[int],
    n_val: int,
    n_test: int,
    noise: float,
    C: float,
    points: tuple[int, ...] = (),
) -> dict[str, Any]:
    """in-dist, LOO и (опционально) кривая few-shot для одного сценария."""
    dim = spec.n_questions
    held = spec.held_out
    in_dist: list[dict[str, Any]] = []
    loo: list[dict[str, Any]] = []
    shots: list[dict[str, Any]] = []

    for seed in seeds:
        rng = np.random.default_rng(seed)
        val = _dataset(spec, n=n_val, rng=rng, noise=noise)
        test = _dataset(spec, n=n_test, rng=rng, noise=noise)
        train = val.subset(val.question != held)
        own_val = val.subset(val.question == held)
        test_held = test.subset(test.question == held)

        for name in VARIANTS:
            passed = _scores(name, val, test, dim=dim, seed=seed, C=C)
            in_dist.append(
                {"variant": name, "seed": seed, **risk_report(passed, test.correct)}
            )
            scored = _scores(name, train, test_held, dim=dim, seed=seed, C=C)
            loo.append(
                {
                    "variant": name,
                    "seed": seed,
                    **risk_report(scored, test_held.correct),
                }
            )
            if points:
                order = rng.permutation(len(own_val.question))
                for point in points:
                    index = order[:point]
                    extra = _Data(
                        own_val.matrix[index],
                        own_val.n_classes[index],
                        own_val.question[index],
                        own_val.correct[index],
                    )
                    train_point = _concat(train, extra) if point else train
                    scored = _scores(
                        name, train_point, test_held, dim=dim, seed=seed, C=C
                    )
                    shots.append(
                        {
                            "variant": name,
                            "shots": point,
                            "seed": seed,
                            **risk_report(scored, test_held.correct),
                        }
                    )

    report: dict[str, Any] = {
        "scenario": spec.name,
        "note": spec.note,
        "drift": spec.drift,
        "seeds": list(seeds),
        "in_dist": _mean_rows(in_dist, ("variant",)),
        "loo": _mean_rows(loo, ("variant",)),
    }
    if points:
        report["shots"] = _mean_rows(shots, ("variant", "shots"))
    return report


def _print_table(title: str, rows: list[dict[str, Any]], keys: tuple[str, ...]) -> None:
    print(f"\n{title}")
    header = "  ".join(f"{key:>10}" for key in keys)
    print(f"{'вариант':>22}  {header}")
    for row in rows:
        cells = "  ".join(f"{row[key]:>10.4f}" for key in keys)
        label = row["variant"]
        if "shots" in row:
            label = f"{label} n={row['shots']}"
        print(f"{label:>22}  {cells}")


DEFAULT_KEYS = ("aurc", "eaurc", "acc_at_0.7_coverage", "ece", "brier")


def _print_scenario(report: dict[str, Any]) -> None:
    print("\n" + "=" * 78)
    print(f"{report['scenario']}: {report['note']}")
    print(f"seeds={report['seeds']}  drift={report['drift']}")
    _print_table(
        "в пуле (обучение и замер на всех вопросах)",
        report["in_dist"],
        DEFAULT_KEYS,
    )
    _print_table(
        "на новом вопросе (обучение без него, замер на нём)",
        report["loo"],
        DEFAULT_KEYS,
    )
    if "shots" in report:
        for point in SHOT_POINTS:
            rows = [r for r in report["shots"] if r["shots"] == point]
            _print_table(
                f"своих строк: {point} (отложенный вопрос)",
                rows,
                ("aurc", "acc_at_0.7_coverage", "ece", "brier"),
            )


def _print_calibration(reports: dict[str, dict[str, Any]]) -> None:
    """S5: ECE/Brier для S1 (in-dist + LOO) и S4 (по точкам few-shot)."""
    print("\n" + "=" * 78)
    print("S5. Калибровка при сдвиге (ECE/Brier, меньше — лучше)")
    s1 = reports.get("S1")
    if s1 is not None:
        _print_table("S1, в пуле", s1["in_dist"], ("ece", "brier"))
        _print_table("S1, на новом вопросе", s1["loo"], ("ece", "brier"))
    s4 = reports.get("S4")
    if s4 is not None and "shots" in s4:
        for point in SHOT_POINTS:
            rows = [r for r in s4["shots"] if r["shots"] == point]
            _print_table(
                f"S4, {point} своих строк (отложенный вопрос)",
                rows,
                ("ece", "brier"),
            )
    print(
        "\nЧитается так: пока сдвиг вопроса неизвестен, оценка риска смещена, и\n"
        "именно калибровка (ECE/Brier) показывает это раньше ранжирования: AURC\n"
        "внутри одного вопроса к постоянному сдвигу нечувствителен."
    )


def build_summary(
    *,
    scenarios: list[str],
    seeds: list[int],
    n_val: int,
    n_test: int,
    noise: float,
    C: float,
) -> dict[str, Any]:
    """Отчёты по сценариям; ``S5`` — отчёт по ``S1``/``S4``, не свои данные."""
    needed = [name for name in dict.fromkeys(scenarios) if name in SCENARIOS]
    if "S5" in scenarios:
        needed += [name for name in ("S1", "S4") if name not in needed]
    reports = {
        name: scenario_report(
            SCENARIOS[name],
            seeds=seeds,
            n_val=n_val,
            n_test=n_test,
            noise=noise,
            C=C,
            points=SHOT_POINTS if name in WITH_SHOTS else (),
        )
        for name in needed
    }
    return {
        "config": {
            "scenarios": scenarios,
            "seeds": seeds,
            "n_val": n_val,
            "n_test": n_test,
            "noise": noise,
            "C": C,
            "feature_columns": list(FEATURE_COLUMNS),
            "variants": list(VARIANTS),
        },
        "reports": reports,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--scenarios",
        nargs="*",
        default=["S1", "S2", "S3", "S4", "S5", "S6"],
        choices=["S1", "S2", "S3", "S4", "S5", "S6"],
        help="какие сценарии считать (S5 — отчёт по S1/S4)",
    )
    parser.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    parser.add_argument(
        "--n-val", type=int, default=300, help="строк валидации на вопрос"
    )
    parser.add_argument(
        "--n-test", type=int, default=1000, help="строк теста на вопрос"
    )
    parser.add_argument(
        "--noise", type=float, default=0.25, help="шум наблюдаемых признаков"
    )
    parser.add_argument("--C", type=float, default=1.0, help="сила L2 логистики")
    parser.add_argument("--out-dir", type=Path, default=Path("data/risk_simulate"))
    args = parser.parse_args(argv)

    summary = build_summary(
        scenarios=args.scenarios,
        seeds=args.seeds,
        n_val=args.n_val,
        n_test=args.n_test,
        noise=args.noise,
        C=args.C,
    )
    for name in dict.fromkeys(args.scenarios):
        if name == "S5":
            continue
        _print_scenario(summary["reports"][name])
    if "S5" in args.scenarios:
        _print_calibration(summary["reports"])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = args.out_dir / "summary.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nsummary -> {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
