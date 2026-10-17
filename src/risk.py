"""Оценка вероятности ошибки решения — обучение отказу (selective prediction).

Каскад может эскалировать не по «сырой» уверенности, а по оценке вероятности
ошибки локального решения. Модель — логистическая регрессия на признаках
решения (уверенность, нормированная энтропия, отрыв лучшего класса, разрыв
логитов двух лучших исходов, число исходов и знаковый логит `noul`;
подробности — в ``decision_features``); обучается на отложенном наборе
(``training.recipe.risk``), а при обслуживании считается на numpy — то есть в
пути инференса нет тяжёлых зависимостей.
"""
from __future__ import annotations

import json
import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.schemas import DecisionResult

FEATURE_NAMES = (
    "confidence",
    "entropy",
    "margin",
    "log_odds_gap",
    "n_classes",
    "noul_logit",
)

#: Состав признаков моделей, обученных до расширения (файлы ``risk.json`` с
#: тремя весами). Такие модели продолжают работать в прежней семантике:
#: ``predict_error`` берёт по именам только те признаки, на которых модель
#: обучена.
LEGACY_FEATURE_NAMES = ("confidence", "entropy", "margin")

#: Сила L2 логистической регрессии риск-модели по умолчанию. ``0.2`` — по замеру
#: Шага 5а (``docs/risk-aware-cascade.md``, «Находка: провал hash — это
#: перерегуляризация»): на RouterBench/hash, RouterBench/gpt2 и AG News/gpt2
#: перенос (LOO) не хуже, in-dist не хуже, разброс по seeds не растёт, а падает.
#: Прежний ``1.0`` переобучал межвопросные константы и на hash значимо проигрывал
#: ``confidence`` на новом вопросе.
RISK_C_DEFAULT = 0.2

#: Минимум размеченных примеров нового вопроса, при котором есть смысл
#: переобучать риск-модель (few-shot, Шаг 6). Замер
#: (``docs/risk-aware-cascade.md``, «Сколько примеров нового вопроса нужно»):
#: десяток своих строк уже снимает основную часть потери переноса, а на
#: RouterBench/gpt2 разница становится значимой к ``n ≈ 50``. Дефолт ``10`` —
#: компромисс «дёшево и заметно»; порог настраивается ``DECISION_RISK_FEW_SHOT_N``.
RISK_FEW_SHOT_MIN_N = 10


def decision_distribution(decision: DecisionResult) -> np.ndarray:
    """Распределение вероятностей решения как вектор (noul -> [p, 1 - p])."""
    if decision.type == "noul":
        p = float(decision.noul)
        return np.asarray([p, 1.0 - p], dtype=float)
    probabilities = getattr(decision, "probabilities", None)
    if probabilities:
        values = np.asarray([float(v) for v in probabilities.values()], dtype=float)
    else:
        values = np.asarray([float(decision.confidence)], dtype=float)
    total = values.sum()
    return values / total if total > 0.0 else values


def decision_features(decision: DecisionResult) -> np.ndarray:
    """Признаки решения для оценщика риска (порядок — ``FEATURE_NAMES``).

    Первые три выведены из нормированного распределения; последние три добавлены
    по замеру ``eval/risk/features.py`` (AURC −3…−31% против трёх признаков,
    см. `docs/risk-aware-cascade.md`):

    - ``log_odds_gap`` — `ln(p_1 / p_2)` двух лучших исходов. Для `k >= 3` это
      **разрыв сырых логитов до softmax** (свойство softmax: `logit_i - logit_j =
      ln(p_i / p_j)`), то есть «сила улики», несводимая к нормированному
      распределению; для `k = 2` детерминированно вытекает из уверенности и
      вклада не даёт.
    - ``n_classes`` — число исходов решения (для `noul` — 2).
    - ``noul_logit`` — знаковый логит `noul`: первые три признака симметричны
      относительно `p <-> 1-p`, поэтому знак теряется.

    Для `noul` распределение — это `[p, 1-p]`.
    """
    dist = decision_distribution(decision)
    order = np.sort(dist)[::-1]
    confidence = float(order[0])
    if dist.size > 1:
        entropy = float(
            -(dist * np.log(np.clip(dist, 1e-12, 1.0))).sum() / math.log(dist.size)
        )
        margin = float(order[0] - order[1])
        log_odds_gap = float(
            np.log(np.clip(order[0], 1e-12, None))
            - np.log(np.clip(order[1], 1e-12, None))
        )
    else:
        entropy = 0.0
        margin = 1.0
        log_odds_gap = 0.0
    noul_logit = 0.0
    if decision.type == "noul":
        p = float(np.clip(decision.noul, 1e-12, 1.0 - 1e-12))
        noul_logit = float(np.log(p / (1.0 - p)))
    return np.asarray(
        [confidence, entropy, margin, log_odds_gap, float(dist.size), noul_logit],
        dtype=float,
    )


def features_matrix(decisions: Iterable[DecisionResult]) -> np.ndarray:
    return np.vstack([decision_features(d) for d in decisions])


def _sigmoid(z: float) -> float:
    if z >= 0.0:
        return 1.0 / (1.0 + math.exp(-z))
    exp_z = math.exp(z)
    return exp_z / (1.0 + exp_z)


def _sigmoid_array(z: np.ndarray) -> np.ndarray:
    out = np.empty_like(z, dtype=float)
    positive = z >= 0.0
    out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    exp_z = np.exp(z[~positive])
    out[~positive] = exp_z / (1.0 + exp_z)
    return out


def _as_matrix(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    if matrix.ndim != 2:
        raise ValueError(f"features must be 1-D or 2-D, got ndim={matrix.ndim}")
    return matrix


def fit_logistic(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    C: float = RISK_C_DEFAULT,
    max_iter: int = 100,
    tol: float = 1e-8,
) -> tuple[np.ndarray, float]:
    """L2-логистическая регрессия на чистом numpy (Ньютон/IRLS), без sklearn.

    Нужна в пути обслуживания: few-shot переобучение риск-модели под новый
    вопрос (Шаг 6) должно быть CPU-only и мгновенным, а тяжёлые зависимости в
    инференсе нежелательны. Соглашение о ``C`` — как у
    ``sklearn.linear_model.LogisticRegression``: минимизируется
    ``0.5·||w||² + C·Σ logloss`` (свободный член не регуляризуется), то есть
    меньше ``C`` — сильнее L2. Возвращает ``(weights, bias)``.

    Если класс в выборке один (все верны/все ошибочны), возвращает нулевые веса
    и базовую вероятность — как ``training.recipe.risk``.
    """
    matrix = _as_matrix(features)
    target = np.asarray(labels, dtype=float).reshape(-1)
    if matrix.shape[0] != target.shape[0]:
        raise ValueError("features rows != labels length")
    if C <= 0.0:
        raise ValueError(f"C must be > 0, got {C}")
    n_features = matrix.shape[1]
    if matrix.shape[0] < 2 or len(np.unique(target)) < 2:
        eps = 1e-6
        p = float(np.clip(target.mean(), eps, 1.0 - eps))
        return np.zeros(n_features), float(math.log(p / (1.0 - p)))

    weights = np.zeros(n_features)
    bias = 0.0
    for _ in range(max_iter):
        probability = _sigmoid_array(matrix @ weights + bias)
        residual = probability - target
        grad = np.concatenate([weights + C * (matrix.T @ residual), [C * residual.sum()]])
        if float(np.max(np.abs(grad))) < tol:
            break
        # вес Ньютона p(1-p); нижняя граница — чтобы шаг не вырождался
        scale = np.clip(probability * (1.0 - probability), 1e-12, None)
        weighted = matrix * np.sqrt(scale)[:, None]
        hessian = np.empty((n_features + 1, n_features + 1))
        hessian[:n_features, :n_features] = (
            np.eye(n_features) + C * (weighted.T @ weighted) + 1e-8 * np.eye(n_features)
        )
        hessian[:n_features, n_features] = C * (matrix.T @ scale)
        hessian[n_features, :n_features] = hessian[:n_features, n_features]
        hessian[n_features, n_features] = C * float(scale.sum()) + 1e-8
        try:
            step = np.linalg.solve(hessian, grad)
        except np.linalg.LinAlgError:
            break
        weights = weights - step[:n_features]
        bias -= float(step[n_features])
    return weights, bias


def gate_verdict(gate: dict[str, Any] | None, delta: float = 0.0) -> tuple[bool, list[str]]:
    """Вердикт hard gate с допуском ``delta``.

    Сертификат несёт точечный вердикт (``accepted``/``worse``), снятый при
    обучении с проектными допусками, и — у сертификатов новых форматов —
    по-метричные превышения ``excess`` (``eval.risk.core.gate_excess``:
    насколько кандидат хуже baseline **сверх** допуска). ``excess`` позволяет
    пересчитать вердикт под дополнительным допуском ``delta``: прод задаёт его
    через ``DECISION_RISK_GATE_DELTA`` (``> 0`` ослабляет гейт, ``< 0``
    ужесточает), не переобучая модель. Без ``excess`` возвращается записанный
    вердикт, и ``delta`` не применяется.

    ``gate`` = ``None`` означает, что замера не было: отказывать нечего —
    ``(True, [])``.
    """
    if not gate:
        return True, []
    excess = gate.get("excess")
    if excess is None:
        return bool(gate.get("accepted", True)), list(gate.get("worse", []))
    worse = [
        f"{key}: превышение {float(value):.4g} > {float(delta):g}"
        for key, value in excess.items()
        if float(value) > float(delta)
    ]
    return len(worse) == 0, worse


@dataclass
class ErrorRiskModel:
    """Логистический оценщик ``P(ошибка)`` по признакам решения.

    ``feature_names`` — имена признаков в том порядке, в каком они идут в
    ``weights``; все должны входить в ``FEATURE_NAMES``. Модели старого формата
    (три признака, ``LEGACY_FEATURE_NAMES``) загружаются и работают как прежде.
    """

    weights: np.ndarray
    bias: float
    feature_names: tuple[str, ...] = FEATURE_NAMES
    #: Заключение hard gate (см. ``eval.risk.core.gate_risk_vs_confidence``):
    #: ``{"accepted": bool, "worse": [...], "tol": ..., ...}``. Модель с явным
    #: отказом гейта в проде не применяется — каскад фолбэчит на порог
    #: уверенности. ``None`` — замеры не проводились, заключения нет.
    gate: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        self.weights = np.asarray(self.weights, dtype=float).reshape(-1)
        if self.weights.size != len(self.feature_names):
            raise ValueError(
                f"weights size {self.weights.size} != "
                f"feature_names {len(self.feature_names)}"
            )
        unknown = [name for name in self.feature_names if name not in FEATURE_NAMES]
        if unknown:
            raise ValueError(
                f"unknown feature_names {unknown}; known: {list(FEATURE_NAMES)}"
            )

    def vector(self, decision: DecisionResult) -> np.ndarray:
        """Вектор признаков решения в порядке ``self.feature_names``."""
        features = dict(zip(FEATURE_NAMES, decision_features(decision)))
        return np.asarray([features[name] for name in self.feature_names],
                          dtype=float)

    def predict_error(self, decision: DecisionResult) -> float:
        """Вероятность того, что решение неверно (0..1)."""
        z = float(np.dot(self.weights, self.vector(decision)) + self.bias)
        return _sigmoid(z)

    def should_escalate(self, decision: DecisionResult, risk_budget: float) -> bool:
        return self.predict_error(decision) > risk_budget

    def retrained_with(
        self,
        own_features: np.ndarray,
        own_labels: np.ndarray,
        *,
        reference_features: np.ndarray | None = None,
        reference_labels: np.ndarray | None = None,
        C: float = RISK_C_DEFAULT,
    ) -> ErrorRiskModel:
        """Переобучить на ``{общие (reference) + свои (own)}`` — few-shot Шага 6.

        ``own_labels``/``reference_labels`` — **корректность** решения (1/True =
        верно), как в ``training.recipe.risk``. Замысел схемы: общие строки не
        дают модели «забыть» общий сигнал, свои снимают перенос коэффициента с
        других вопросов (``docs/risk-aware-cascade.md``, «Честный вывод»).

        Состав признаков сохраняется (``self.feature_names``), сертификат гейта
        — нет: это другая модель, её вердикт берётся отдельно (инференс-гейт
        каскада остаётся жёстким).
        """
        matrices = [_as_matrix(own_features)]
        targets = [np.asarray(own_labels, dtype=float).reshape(-1)]
        if reference_features is not None or reference_labels is not None:
            if reference_features is None or reference_labels is None:
                raise ValueError("reference features and labels must go together")
            matrices.append(_as_matrix(reference_features))
            targets.append(np.asarray(reference_labels, dtype=float).reshape(-1))
        features = np.vstack(matrices)
        if features.shape[1] == len(FEATURE_NAMES):
            # вход в порядке FEATURE_NAMES (обычно из ``features_matrix``):
            # для модели старого формата оставляем только её признаки
            columns = [FEATURE_NAMES.index(name) for name in self.feature_names]
            features = features[:, columns]
        elif features.shape[1] != len(self.feature_names):
            raise ValueError(
                f"features cols {features.shape[1]} не совпадают ни с "
                f"FEATURE_NAMES ({len(FEATURE_NAMES)}), ни с feature_names "
                f"модели ({len(self.feature_names)})"
            )
        errors = 1.0 - np.concatenate(targets)
        weights, bias = fit_logistic(features, errors, C=C)
        return ErrorRiskModel(
            weights=weights, bias=bias, feature_names=self.feature_names
        )

    def gate_verdict(self, delta: float = 0.0) -> tuple[bool, list[str]]:
        """Вердикт hard gate с допуском ``delta`` (см. ``gate_verdict`` модуля)."""
        return gate_verdict(self.gate, delta)

    @property
    def gate_rejected(self) -> bool:
        """Известен ли отказ hard gate: замера не было — не отказ.

        Модель считается неприемлемой только при **явном** вердикте
        «не лучше baseline»: откатывать нечего, пока ничего не измеряли.
        """
        return not gate_verdict(self.gate, 0.0)[0]

    def to_dict(self) -> dict:
        payload = {
            "feature_names": list(self.feature_names),
            "weights": [float(w) for w in self.weights],
            "bias": float(self.bias),
        }
        if self.gate is not None:
            payload["gate"] = self.gate
        return payload

    @classmethod
    def from_dict(cls, payload: dict) -> ErrorRiskModel:
        weights = np.asarray(payload["weights"], dtype=float)
        names = payload.get("feature_names")
        if names is None:
            # файл без списка признаков: определяем состав по числу весов, иначе
            # трёхвесовый ``risk.json`` из старого формата не загрузится
            names = (
                LEGACY_FEATURE_NAMES
                if weights.size == len(LEGACY_FEATURE_NAMES)
                else FEATURE_NAMES
            )
        return cls(
            weights=weights,
            bias=float(payload["bias"]),
            feature_names=tuple(names),
            gate=payload.get("gate"),
        )

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> ErrorRiskModel:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
