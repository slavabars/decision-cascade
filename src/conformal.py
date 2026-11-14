"""Selective prediction с distribution-free гарантией доли ошибок.

Обычный бюджет риска задаёт *желаемую* долю ошибок, но не говорит, с какой
вероятностью она выдержит на новых данных: порог подобран на конечной выборке,
и на коротком срезе эмпирическая доля ошибок гуляет. Сертификат закрывает этот
пробел: по отложенной калибровочной выборке выбирается порог на ``P(ошибка)``,
для которого **верхняя граница Клоппера–Пирсона** доли ошибок среди принятых
решений не превышает ``alpha``. Тогда с вероятностью не меньше ``1 - delta``
истинная доля ошибок среди принятых не выше ``alpha`` — без предположений о
модели, только обмениваемость калибровочной и новых выборок.

Порог берётся максимально широкий (максимальное покрытие), при котором
граница ещё укладывается в ``alpha``. Если выборки не хватает, сертификат
честно отдаёт покрытие 0: лучше не принять ничего, чем пообещать больше, чем
данные позволяют.

Сертификат — файл ``conformal.json``: порог, покрытие, эмпирический риск,
верхняя граница и вердикт гейта «не навреди» на отложенном тесте. Прод читает
готовый порог (``src.backends.conformal``) и не считает статистику в пути
инференса.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

#: Доля ошибок среди принятых решений, которую обещает сертификат.
DEFAULT_ALPHA = 0.05
#: Допустимая вероятность нарушить обещание (уровень доверия ``1 - delta``).
#: 0.1 — практичный дефолт: при делении на вопросы (Бонферрони) 0.05 требует
#: слишком много калибровочных строк на каждый вопрос.
DEFAULT_DELTA = 0.1
#: Минимальное покрытие, ниже которого сертификат считается бесполезным.
DEFAULT_MIN_COVERAGE = 0.3
SCHEMA_VERSION = "1"
METHOD = "split_conformal_clopper_pearson"


def clopper_pearson_upper(errors: int, total: int, delta: float) -> float:
    """Верхняя граница доли ошибок при ``total`` испытаниях и ``errors`` отказах.

    Точная граница Клоппера–Пирсона: квантиль ``Beta(errors + 1, total - errors)``
    уровня ``1 - delta``. Для ``errors = 0`` это ``1 - delta ** (1 / total)``,
    поэтому нулевая ошибка на десятке примеров ничего не доказывает: чтобы
    сертифицировать ``alpha = 0.05`` при ``delta = 0.05``, нужно около 59
    примеров без единой ошибки.
    """
    if total < 0 or errors < 0 or errors > total:
        raise ValueError(f"need 0 <= errors <= total, got {errors}/{total}")
    if not 0.0 < delta < 1.0:
        raise ValueError(f"delta must be in (0, 1), got {delta}")
    if total == 0 or errors == total:
        return 1.0
    from scipy.stats import beta

    return float(beta.ppf(1.0 - delta, errors + 1, total - errors))


def _as_scores(scores: Any, correct: Any) -> tuple[np.ndarray, np.ndarray]:
    scores = np.asarray(scores, dtype=float).reshape(-1)
    correct = np.asarray(correct, dtype=float).reshape(-1)
    if scores.shape != correct.shape:
        raise ValueError(f"scores {scores.shape} != correct {correct.shape}")
    if scores.size == 0:
        raise ValueError("need at least one example")
    if not np.all((scores >= 0.0) & (scores <= 1.0)):
        raise ValueError("scores are P(error) and must be in [0, 1]")
    if not np.all((correct == 0.0) | (correct == 1.0)):
        raise ValueError("correct must be 0/1")
    return scores, correct


def selective_certificate(
    scores: Any,
    correct: Any,
    *,
    groups: Any | None = None,
    alpha: float = DEFAULT_ALPHA,
    delta: float = DEFAULT_DELTA,
    min_coverage: float = DEFAULT_MIN_COVERAGE,
) -> dict[str, Any]:
    """Порог на ``P(ошибка)`` с гарантией доли ошибок среди принятых.

    ``scores`` — оценка ``P(ошибка)`` (больше — хуже), ``correct`` — факт
    корректности (1/0). Решение принимается локально, если
    ``score <= threshold``. Порог — максимальное покрытие, при котором верхняя
    граница Клоппера–Пирсона не превышает ``alpha``. Возвращает сертификат,
    готовый к записи в ``conformal.json`` (см. ``save_certificate``).

    ``groups`` — необязательные метки вопросов. Если заданы, гарантия даётся
    **по каждому вопросу одновременно**: уровень ``delta`` делится между
    группами (неравенство Бонферрони), и порог должен проходить во всех.
    Решения разных вопросов на одном состоянии коррелированы, поэтому
    складывать их в одну выборку и обещать независимость нельзя; по-вопросная
    гарантия этого не требует.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if not 0.0 < delta < 1.0:
        raise ValueError(f"delta must be in (0, 1), got {delta}")
    if not 0.0 <= min_coverage <= 1.0:
        raise ValueError(f"min_coverage must be in [0, 1], got {min_coverage}")
    scores, correct = _as_scores(scores, correct)
    n = scores.size
    if groups is None:
        group_labels = np.full(n, "all", dtype=object)
    else:
        group_labels = np.asarray(groups, dtype=object).reshape(-1)
        if group_labels.size != n:
            raise ValueError(
                f"groups {group_labels.shape} != scores {scores.shape}"
            )
    unique_groups = sorted({str(label) for label in group_labels})
    group_masks = {
        name: group_labels == name for name in unique_groups
    }
    per_delta = delta / len(unique_groups)

    order = np.argsort(scores, kind="stable")
    sorted_scores = scores[order]
    errors = (1.0 - correct)[order]
    sorted_groups = group_labels[order]

    # Кумулятивные счётчики по вопросам: для каждого кандидата-порога
    # принятые/ошибочные решения группы читаются за O(1), а не пересчитываются
    # заново. Иначе подбор порога на 10k решений становится квадратичным.
    code = {name: index for index, name in enumerate(unique_groups)}
    codes = np.asarray([code[str(label)] for label in sorted_groups], dtype=int)
    one_hot = np.zeros((len(unique_groups), n), dtype=np.int64)
    one_hot[codes, np.arange(n)] = 1
    cumulative_accepted = np.cumsum(one_hot, axis=1)
    cumulative_errors = np.cumsum(one_hot * errors.astype(np.int64), axis=1)
    group_sizes = {name: int(mask.sum()) for name, mask in group_masks.items()}

    best: dict[str, Any] | None = None
    for index in range(n):
        # рассматриваем только конец группы одинаковых score: порог не может
        # принять часть решений с тем же P(ошибка) и отвергнуть другую
        if index + 1 < n and sorted_scores[index] == sorted_scores[index + 1]:
            continue
        accepted = index + 1
        coverage = accepted / n
        if coverage < min_coverage:
            continue
        group_stats = _group_stats_at(
            cumulative_accepted[:, index],
            cumulative_errors[:, index],
            unique_groups,
            group_sizes,
            per_delta,
        )
        if all(
            stats["upper_bound"] <= alpha for stats in group_stats.values()
        ):
            best = {
                "threshold": float(sorted_scores[index]),
                "accepted": accepted,
                "coverage": float(coverage),
                "empirical_risk": float(errors[:accepted].mean()),
                "upper_bound": max(
                    stats["upper_bound"] for stats in group_stats.values()
                ),
                "groups": group_stats,
            }

    if best is None:
        best = {
            "threshold": -1.0,
            "accepted": 0,
            "coverage": 0.0,
            "empirical_risk": None,
            "upper_bound": 1.0,
            "groups": {
                name: {
                    "n": group_sizes[name],
                    "accepted": 0,
                    "coverage": 0.0,
                    "empirical_risk": None,
                    "upper_bound": 1.0,
                    "delta": per_delta,
                }
                for name in unique_groups
            },
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "method": METHOD,
        "score": "p_error",
        "alpha": float(alpha),
        "delta": float(delta),
        "n": int(n),
        "min_coverage": float(min_coverage),
        "gate": None,
        **best,
    }


def _group_stats_at(
    accepted_counts: np.ndarray,
    error_counts: np.ndarray,
    unique_groups: list[str],
    group_sizes: dict[str, int],
    delta: float,
) -> dict[str, dict[str, Any]]:
    """Статистика групп при текущем пороге (кумулятивные счётчики)."""
    stats: dict[str, dict[str, Any]] = {}
    for position, name in enumerate(unique_groups):
        accepted = int(accepted_counts[position])
        if accepted == 0:
            stats[name] = {
                "n": group_sizes[name],
                "accepted": 0,
                "coverage": 0.0,
                "empirical_risk": None,
                "upper_bound": 1.0,
                "delta": delta,
            }
            continue
        n_errors = int(error_counts[position])
        stats[name] = {
            "n": group_sizes[name],
            "accepted": accepted,
            "coverage": accepted / group_sizes[name],
            "empirical_risk": n_errors / accepted,
            "upper_bound": clopper_pearson_upper(n_errors, accepted, delta),
            "delta": delta,
        }
    return stats


def validate_certificate(certificate: dict[str, Any]) -> dict[str, Any]:
    """Проверить форму сертификата и вернуть его же (для чтения с диска)."""
    if not isinstance(certificate, dict):
        raise TypeError("conformal certificate must be a JSON object")
    if certificate.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            "conformal certificate schema_version must be "
            f"{SCHEMA_VERSION!r}, got {certificate.get('schema_version')!r}"
        )
    if certificate.get("method") != METHOD:
        raise ValueError(
            f"conformal certificate method must be {METHOD!r}, "
            f"got {certificate.get('method')!r}"
        )
    for key in ("alpha", "delta"):
        value = float(certificate.get(key))
        if not 0.0 < value < 1.0:
            raise ValueError(f"certificate {key} must be in (0, 1), got {value}")
    threshold = float(certificate.get("threshold"))
    if not -1.0 <= threshold <= 1.0:
        raise ValueError(
            f"certificate threshold must be in [-1, 1], got {threshold}"
        )
    coverage = float(certificate.get("coverage"))
    if not 0.0 <= coverage <= 1.0:
        raise ValueError(
            f"certificate coverage must be in [0, 1], got {coverage}"
        )
    gate = certificate.get("gate")
    if gate is not None and not isinstance(gate, dict):
        raise TypeError("certificate gate must be an object or null")
    groups = certificate.get("groups")
    if groups is not None and not isinstance(groups, dict):
        raise TypeError("certificate groups must be an object or null")
    return certificate


def load_certificate(path: str | Path) -> dict[str, Any]:
    """Прочитать и проверить ``conformal.json``."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return validate_certificate(payload)


def save_certificate(certificate: dict[str, Any], path: str | Path) -> Path:
    """Записать сертификат (форма проверяется перед записью)."""
    validate_certificate(certificate)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(certificate, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path
