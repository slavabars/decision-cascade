"""Общие хелперы для примеров: сервис, вопросы, печать и сериализация.

Примеры лежат в подпапках и подключают модуль как ``examples._common``: сначала
корень репозитория добавляется в ``sys.path``, затем импортируется этот модуль.
Здесь собрано всё, что иначе повторялось бы из примера в пример: построение
in-process сервиса (по умолчанию — без загрузки моделей), готовые наборы
вопросов, форматирование и JSON/CSV-сериализация типизированных решений.
"""
from __future__ import annotations

import csv
import dataclasses
import json
from pathlib import Path
from typing import Any

from api.server.service import DecisionService, HashStateEncoder
from eval.datasets.public_dataset import PUBLIC_QUESTIONS, PUBLIC_SPECS
from eval.datasets.text_dataset import (
    QUESTIONS,
    SPECS,
    generate_examples,
    spec_from_api,
)
from src.encoder.vision import HashVisionEncoder

REPO_ROOT = Path(__file__).resolve().parents[1]

TYPE_LABEL = {"noul": "Бинарный", "choice": "Выбор", "score": "Шкала"}

UNTRAINED_NOTE = (
    "Примечание: головы необучены (hash-энкодер) — это демонстрация конвейера, "
    "а не качества решений. Обучите головы или задайте обученный пак весов."
)


# ---------------------------------------------------------------------------
# Вопросы
# ---------------------------------------------------------------------------
def ticket_questions() -> dict[str, dict[str, Any]]:
    """Шесть вопросов поддержки (API-формат: noul/choice/score)."""
    return QUESTIONS


def ticket_specs() -> dict[str, Any]:
    """Те же вопросы, но во внутренних спецификациях голов."""
    return SPECS


def public_questions() -> dict[str, dict[str, Any]]:
    """Вопросы публичного набора AG News (topic/business/tech_level)."""
    return PUBLIC_QUESTIONS


def public_specs() -> dict[str, Any]:
    """Вопросы AG News во внутренних спецификациях."""
    return PUBLIC_SPECS


def specs_from_api(questions: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """API-описание вопросов -> спецификации голов (для in-process прогонов)."""
    return {name: spec_from_api(data) for name, data in questions.items()}


def sample_tickets(n: int, seed: int = 0) -> list[dict[str, Any]]:
    """Синтетические обращения с метками — быстрый материал для примеров."""
    return generate_examples(n, seed=seed)


# ---------------------------------------------------------------------------
# Сервис
# ---------------------------------------------------------------------------
def build_service(
    *,
    seed: int = 0,
    dim: int = 64,
    checkpoint_dir: str | Path | None = None,
    vision: bool = False,
    vision_dim: int = 64,
    model_id: str | None = None,
    encoder_kind: str | None = None,
    feature_dim: int | None = None,
    device: str | None = None,
    dtype: str | None = None,
    max_length: int | None = 256,
) -> DecisionService:
    """Собирает in-process сервис.

    Без ``model_id`` используется детерминированный hash-энкодер: примеры
    запускаются мгновенно и без скачивания моделей. С ``model_id`` —
    настоящий трансформер (нужны веса на HF Hub и, возможно, GPU).
    """
    vision_encoder = HashVisionEncoder(dim=vision_dim) if vision else None
    if not model_id:
        return DecisionService(
            encoder=HashStateEncoder(dim),
            dim=dim,
            seed=seed,
            checkpoint_dir=checkpoint_dir,
            vision_encoder=vision_encoder,
        )

    from src.device import resolve_device, resolve_dtype
    from src.encoder.factory import create_state_encoder

    resolved_dim = feature_dim or 768
    encoder = create_state_encoder(
        model_id,
        resolved_dim,
        kind=encoder_kind,
        dtype=resolve_dtype(dtype),
        device=resolve_device(device),
        max_length=max_length,
    )
    encoder.load()
    return DecisionService(
        encoder=encoder,
        dim=resolved_dim,
        seed=seed,
        checkpoint_dir=checkpoint_dir,
        vision_encoder=vision_encoder,
    )


# ---------------------------------------------------------------------------
# Печать и сериализация решений
# ---------------------------------------------------------------------------
def decision_to_dict(decision: Any) -> dict[str, Any]:
    """Типизированное решение (dataclass или dict) -> обычный dict."""
    if dataclasses.is_dataclass(decision) and not isinstance(decision, type):
        data = dataclasses.asdict(decision)
    else:
        data = dict(decision)
    legend = data.get("legend")
    if isinstance(legend, tuple):
        data["legend"] = list(legend)
    return data


def format_decision(name: str, decision: Any) -> str:
    """Человекочитаемая строка по одному решению."""
    data = decision_to_dict(decision)
    kind = data.get("type")
    label = TYPE_LABEL.get(kind, kind or "?")
    lines = [f"— {name} —  [{label}]"]
    if kind == "noul":
        lines.append(
            f"    значение: {data.get('noul', 0.0):.3f}   "
            f"уверенность: {data.get('confidence', 0.0):.3f}"
        )
    elif kind == "choice":
        lines.append(
            f"    выбор: {data.get('choice')!r}   "
            f"уверенность: {data.get('confidence', 0.0):.3f}"
        )
        for option, prob in _sorted_probs(data.get("probabilities", {})):
            lines.append(f"      {option:<22} {prob:.3f}")
    elif kind == "score":
        lines.append(f"    балл: {data.get('score', 0.0):.3f}")
        for level, prob in _sorted_probs(data.get("probabilities", {})):
            lines.append(f"      {level:<22} {prob:.3f}")
    return "\n".join(lines)


def print_decisions(decisions: dict[str, Any], *, title: str | None = None) -> None:
    """Печатает все решения блока."""
    if title:
        print(title)
    for name, decision in decisions.items():
        print(format_decision(name, decision))


def _sorted_probs(probabilities: dict[str, float]) -> list[tuple[str, float]]:
    return sorted(probabilities.items(), key=lambda kv: -kv[1])


def primary_value(decisions: dict[str, Any], name: str) -> Any:
    """Главное значение решения: вероятность noul, выбор или балл.

    Возвращает ``None``, если вопроса нет в ответе. Удобно для таблиц и CSV.
    """
    decision = decisions.get(name)
    if decision is None:
        return None
    data = decision_to_dict(decision)
    if data.get("type") == "noul":
        return data.get("noul")
    if data.get("type") == "choice":
        return data.get("choice")
    return data.get("score")


# ---------------------------------------------------------------------------
# Файлы: JSONL и CSV
# ---------------------------------------------------------------------------
def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def read_csv_rows(
    path: str | Path, *, text_column: str
) -> list[dict[str, str]]:
    """Читает CSV и проверяет наличие колонки с текстом."""
    with Path(path).open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or text_column not in reader.fieldnames:
            raise ValueError(
                f"CSV must have a {text_column!r} column, got {reader.fieldnames}"
            )
        return [dict(row) for row in reader]


def write_csv_rows(
    path: str | Path, rows: list[dict[str, Any]], fieldnames: list[str]
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path
