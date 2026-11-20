"""Долговременное хранение обратной связи: JSONL-журнал на диске.

Без журнала few-shot примеры risk-каскада и счётчики benefit-каскада живут
только в памяти процесса и обнуляются при рестарте: клиент шлёт
``POST /v1/feedback``, получает ``200``, а после перезапуска сервис снова
опирается на срез, снятый при развёртывании.

Формат — JSONL, по строке на событие. Каждая строка несёт ``v`` (версия схемы)
и ``kind``: ``risk`` — размеченное решение (``question``, ``decision``,
``correct``), ``benefit`` — исход ответа внешней модели (``question``,
``correct``). Журнал только дописывается, а когда строк становится больше
предела, переписывается целиком через временный файл и ``os.replace``: читатель
никогда не видит половину файла.

Предел — по числу событий, а не по размеру: старые события вытесняются, потому
что свежая обратная связь важнее давней, а размер памяти и время переобучения
должны быть ограничены. Один файл на процесс: параллельная запись из нескольких
процессов не поддерживается (это свойство развёртывания, а не журнала), внутри
процесса доступ закрывают блокировки бэкендов.
"""
from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from src.schemas import ChoiceResult, NoulResult, ScoreResult

logger = logging.getLogger(__name__)

#: Версия схемы записи. Строки с другой версией при чтении пропускаются.
SCHEMA_VERSION = 1

#: Тип события: ``risk`` — пример для few-shot риск-модели, ``benefit`` — исход
#: ответа внешней модели для поправки ``q``.
KIND_RISK = "risk"
KIND_BENEFIT = "benefit"
KINDS = (KIND_RISK, KIND_BENEFIT)

#: Сколько событий журнал держит на диске по умолчанию. Тысячи событий — это
#: единицы мегабайт; при переполнении старые вытесняются.
RECORDS_LIMIT = 10_000


def decision_to_record(decision: Any) -> dict[str, Any]:
    """Решение -> JSON-совместимый словарь (форма ``NoulResult``/``ChoiceResult``)."""
    kind = decision.type
    record: dict[str, Any] = {
        "type": kind,
        "confidence": float(decision.confidence),
    }
    if kind == "noul":
        record["noul"] = float(decision.noul)
    elif kind == "choice":
        record["choice"] = str(decision.choice)
        probabilities = getattr(decision, "probabilities", None)
        if probabilities:
            record["probabilities"] = {
                str(name): float(value) for name, value in probabilities.items()
            }
    elif kind == "score":
        record["score"] = float(decision.score)
        probabilities = getattr(decision, "probabilities", None)
        if probabilities:
            record["probabilities"] = {
                str(name): float(value) for name, value in probabilities.items()
            }
        legend = getattr(decision, "legend", ())
        if legend:
            record["legend"] = [str(item) for item in legend]
    else:  # pragma: no cover - тип проверен схемой, ветка на будущее
        raise ValueError(f"unknown decision type: {kind!r}")
    return record


def record_to_decision(payload: Any) -> Any:
    """Словарь из журнала -> типизированное решение.

    Проверяет то, на что опираются признаки риска и отпечаток решения: тип,
    диапазон ``noul``/``score``/``confidence`` и числовые вероятности. Битая
    запись должна падать здесь, а не превращаться в ``nan`` внутри риск-модели.
    """
    if not isinstance(payload, dict):
        raise TypeError("decision must be an object")
    kind = payload.get("type")
    confidence = float(payload.get("confidence", 0.0))
    if not 0.0 <= confidence <= 1.0:
        raise ValueError(f"confidence must be in [0, 1], got {confidence}")
    if kind == "noul":
        noul = float(payload["noul"])
        if not 0.0 <= noul <= 1.0:
            raise ValueError(f"noul must be in [0, 1], got {noul}")
        return NoulResult(noul=noul, confidence=confidence)
    if kind == "choice":
        choice = payload["choice"]
        if not isinstance(choice, str) or not choice:
            raise ValueError("choice must be a non-empty string")
        return ChoiceResult(
            choice=choice,
            probabilities=_probabilities(payload.get("probabilities")),
            confidence=confidence,
        )
    if kind == "score":
        score = float(payload["score"])
        if not 0.0 <= score <= 1.0:
            raise ValueError(f"score must be in [0, 1], got {score}")
        legend = payload.get("legend") or ()
        if not isinstance(legend, (list, tuple)):
            raise ValueError("legend must be a list of strings")
        return ScoreResult(
            score=score,
            probabilities=_probabilities(payload.get("probabilities")),
            confidence=confidence,
            legend=tuple(str(item) for item in legend),
        )
    raise ValueError(f"unknown decision type: {kind!r}")


def _probabilities(value: Any) -> dict[str, float] | None:
    if value is None:
        return None
    if not isinstance(value, dict) or not value:
        raise ValueError("probabilities must be a non-empty object or null")
    result = {}
    for name, raw in value.items():
        probability = float(raw)
        if probability < 0.0:
            raise ValueError(f"probability must be non-negative, got {probability}")
        result[str(name)] = probability
    return result


class FeedbackStore:
    """Журнал событий обратной связи: дописывание, чтение, компакция.

    ``limit`` ограничивает число хранимых событий: при переполнении старые
    вытесняются, а файл переписывается атомарно (``tmp`` + ``os.replace``).
    Каталог журнала создаётся при инициализации — недоступный путь должен
    ломать старт, а не первый же запрос фидбэка.
    """

    def __init__(self, path: str | Path, *, limit: int = RECORDS_LIMIT):
        self.path = Path(path).expanduser()
        self.limit = int(limit)
        if self.limit < 1:
            raise ValueError(f"limit must be >= 1, got {limit}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._records: list[dict[str, Any]] = []
        self._loaded = False
        self._written = 0

    @property
    def records_written(self) -> int:
        """Сколько событий сервис записал за свою жизнь (для диагностики)."""
        return self._written

    def load(self) -> list[dict[str, Any]]:
        """Прочитать журнал (один раз за жизнь объекта) и вернуть копию событий."""
        if not self._loaded:
            self._records = self._read()
            self._loaded = True
            self._written = len(self._records)
        return list(self._records)

    def append(self, records: Iterable[dict[str, Any]]) -> None:
        """Дописать пачку событий и сбросить её на диск (``fsync``)."""
        batch = [self._checked(record) for record in records]
        if not batch:
            return
        self.load()
        payload = "".join(
            json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            for record in batch
        )
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        self._records.extend(batch)
        self._written += len(batch)
        if len(self._records) > self.limit:
            del self._records[: len(self._records) - self.limit]
            self._rewrite()

    def _checked(self, record: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(record, dict):
            raise TypeError("feedback record must be an object")
        kind = record.get("kind")
        if kind not in KINDS:
            raise ValueError(f"feedback record kind must be one of {KINDS}, got {kind!r}")
        checked = dict(record)
        checked["v"] = SCHEMA_VERSION
        return checked

    def _read(self) -> list[dict[str, Any]]:
        try:
            text = self.path.read_text(encoding="utf-8", errors="replace")
        except FileNotFoundError:
            return []
        records: list[dict[str, Any]] = []
        skipped = 0
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue
            if (
                not isinstance(record, dict)
                or record.get("v") != SCHEMA_VERSION
                or record.get("kind") not in KINDS
            ):
                skipped += 1
                continue
            records.append(record)
        if skipped:
            # Содержимое строк не логируем: это тела фидбэка (решения и метки).
            logger.warning(
                "feedback: пропущено %d битых строк в %s", skipped, self.path
            )
        if len(records) > self.limit:
            records = records[-self.limit :]
        return records

    def _rewrite(self) -> None:
        """Атомарно переписать журнал текущим содержимым ``_records``."""
        tmp = self.path.with_name(f"{self.path.name}.{os.getpid()}.tmp")
        try:
            with tmp.open("w", encoding="utf-8") as handle:
                for record in self._records:
                    handle.write(
                        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
                    )
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, self.path)
        finally:
            if tmp.exists():
                tmp.unlink()
        self._written = len(self._records)


def group_records(
    records: Sequence[dict[str, Any]], kind: str
) -> list[dict[str, Any]]:
    """Отобрать события одного типа, не падая на записях другого бэкенда."""
    return [record for record in records if record.get("kind") == kind]
