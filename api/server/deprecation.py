"""Политика депрекации: реестр устаревших поверхностей и заголовки ответа.

Правило (``docs/stability.md`` §5): поверхность объявляется устаревшей минимум
за две минорные версии до удаления. Всё это время она работает, отдаёт
заголовки ``Deprecation``/``Sunset``/``Link`` и один раз пишет предупреждение в
лог. Удаление — только в major-релизе.

Реестр намеренно пуст: пока в проекте нет устаревших поверхностей. Механизм
здесь, чтобы первая же депрекация была объявлена одинаково и проверялась
тестом, а не «на словах».
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger("decision.deprecation")

#: Сколько минорных релизов поверхность обязана прожить после объявления.
MIN_MINOR_RELEASES = 2


def parse_minor(version: str) -> tuple[int, int]:
    """Разбирает ``X.Y[.Z]`` в ``(major, minor)``; иначе ``ValueError``."""
    parts = version.strip().split(".")
    if len(parts) < 2:
        raise ValueError(f"expected MAJOR.MINOR[.PATCH], got {version!r}")
    try:
        return int(parts[0]), int(parts[1])
    except ValueError as exc:
        raise ValueError(f"expected numeric MAJOR.MINOR, got {version!r}") from exc


def minor_distance(since: str, remove_in: str) -> int:
    """Сколько минорных версий между объявлением и удалением."""
    since_major, since_minor = parse_minor(since)
    remove_major, remove_minor = parse_minor(remove_in)
    return (remove_major - since_major) * 1000 + (remove_minor - since_minor)


@dataclass(frozen=True)
class Deprecation:
    """Одна устаревшая поверхность: путь, поле ответа или переменная окружения."""

    #: что именно устарело — путь (``/decisions``) или имя (``DECISION_FOO``)
    item: str
    #: версия, в которой объявлена депрекация
    since: str
    #: версия, в которой поверхность исчезнет (не раньше since + 2 минорные)
    remove_in: str
    #: чем заменить; попадает в ``Link`` и в предупреждение
    alternative: str | None = None
    #: дата отключения в формате RFC 8594 (``Sunset``), необязательна
    sunset: str | None = None

    def headers(self) -> dict[str, str]:
        """Заголовки ответа по RFC 8594 и RFC 9745."""
        headers = {"Deprecation": "true"}
        if self.sunset:
            headers["Sunset"] = self.sunset
        if self.alternative:
            headers["Link"] = f'<{self.alternative}>; rel="deprecation"'
        return headers

    def __post_init__(self) -> None:
        distance = minor_distance(self.since, self.remove_in)
        if distance < MIN_MINOR_RELEASES:
            raise ValueError(
                f"deprecation {self.item!r}: remove_in={self.remove_in} is only "
                f"{distance} minor release(s) after since={self.since}; "
                f"minimum is {MIN_MINOR_RELEASES}"
            )


#: Реестр устаревших поверхностей: ключ — ``Deprecation.item``.
DEPRECATIONS: dict[str, Deprecation] = {}

#: Уже предупреждённые элементы: в лог пишем один раз за жизнь процесса.
_warned: set[str] = set()


def deprecation_for(item: str) -> Deprecation | None:
    """Возвращает депрекацию для поверхности или ``None``."""
    return DEPRECATIONS.get(item)


def warn_once(item: str) -> Deprecation | None:
    """Пишет предупреждение о депрекации один раз и возвращает её запись."""
    deprecation = DEPRECATIONS.get(item)
    if deprecation is None or item in _warned:
        return deprecation
    _warned.add(item)
    logger.warning(
        "deprecated %s since %s, removal in %s: %s",
        deprecation.item,
        deprecation.since,
        deprecation.remove_in,
        deprecation.alternative or "no replacement announced",
    )
    return deprecation


def reset_warnings() -> None:
    """Сбрасывает память предупреждений (для тестов и перезапуска)."""
    _warned.clear()


def validate_registry() -> None:
    """Проверяет, что реестр не нарушает правило двух минорных версий."""
    for item, deprecation in DEPRECATIONS.items():
        if item != deprecation.item:
            raise ValueError(
                f"registry key {item!r} does not match item {deprecation.item!r}"
            )
