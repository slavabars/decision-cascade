"""Политика версионирования схемы решений и миграции артефактов.

Версии — ``<major>[.<minor>]``. Совместимость определяется по **major**:
потребитель, ожидающий ``1.x``, принимает любой ``1.y`` (аддитивные изменения),
но не ``2.*`` (ломающие). Отсутствие версии трактуется как legacy ``1.0`` —
артефакты, созданные до введения версионирования.

Правила эволюции:

* **minor** — добавление необязательных полей/вопросов; старые потребители
  продолжают работать.
* **major** — удаление/переименование полей, смена семантики (например,
  единиц score); требует миграции.

Здесь же живёт версия формата *чекпоинта голов* (``CHECKPOINT_VERSION``),
отдельная от версии ответа API (``src.schemas.SCHEMA_VERSION``).
"""
from __future__ import annotations

from typing import Any

CHECKPOINT_VERSION = "1"


def parse_version(version: str) -> tuple[int, int]:
    """Разбирает ``"<major>[.<minor>]"`` в ``(major, minor)``."""
    text = str(version).strip()
    if not text:
        raise ValueError("version must not be empty")
    parts = text.split(".")
    if len(parts) > 2 or not all(part.isdigit() for part in parts):
        raise ValueError(f"malformed version {version!r}; expected '<major>[.<minor>]'")
    major = int(parts[0])
    minor = int(parts[1]) if len(parts) == 2 else 0
    return major, minor


def version_major(version: str) -> int:
    return parse_version(version)[0]


def is_compatible(expected: str, actual: str | None) -> bool:
    """Совместимы ли версии: одинаковый major. ``None`` считается legacy-совместимым.

    Некорректная строка версии считается несовместимой (а не бросает исключение),
    чтобы предикат оставался безопасным.
    """
    if actual is None:
        return True
    try:
        return version_major(expected) == version_major(actual)
    except ValueError:
        return False


def migrate_checkpoint(payload: dict[str, Any]) -> dict[str, Any]:
    """Приводит payload чекпоинта к текущему формату (или падает явно).

    Legacy-чекпоинты (без ``schema_version``) считаются версией ``1``. Более
    новый major отклонить: старый код не должен молча читать неизвестный формат.
    """
    version = payload.get("schema_version")
    if version is None:
        payload = {**payload, "schema_version": CHECKPOINT_VERSION}
    elif not is_compatible(CHECKPOINT_VERSION, version):
        raise ValueError(
            f"unsupported checkpoint schema_version {version!r}; "
            f"this build understands major {version_major(CHECKPOINT_VERSION)}"
        )
    return payload
