"""Наблюдаемость: request id, структурированные логи, доступ к метрикам.

Контракт:

- каждый ответ несёт ``X-Request-ID``; если клиент прислал свой валидный
  заголовок, он сохраняется (удобно связывать логи нескольких сервисов);
- одна строка лога на запрос (``decision.access``) с методом, путём, кодом,
  длительностью и request id; тела запросов туда не попадают;
- ``DECISION_LOG_FORMAT=json`` включает JSON-логи (для Loki/ELK/journald),
  ``text`` или пусто — обычный формат Python/uvicorn;
- ``DECISION_LOG_LEVEL`` задаёт уровень (по умолчанию ``INFO``).

Request id хранится в ``ContextVar``: синхронные эндпоинты выполняются в
threadpool, но контекст копируется в поток, поэтому id доступен и в логах
сервиса, и в логе доступа.
"""
from __future__ import annotations

import json
import logging
import os
import re
import uuid
from contextvars import ContextVar

REQUEST_ID_HEADER = "X-Request-ID"
LOG_FORMATS = ("text", "json")
#: Допустимые клиентские id: печатаемые, без пробелов, разумной длины.
_VALID_REQUEST_ID = re.compile(r"^[A-Za-z0-9._:@+-]{1,128}$")

_request_id: ContextVar[str | None] = ContextVar("decision_request_id", default=None)


def new_request_id(raw: str | None = None) -> str:
    """Возвращает клиентский request id, если он безопасен, иначе новый UUID4.

    Значение уходит в заголовок ответа и в логи, поэтому произвольную строку
    (переводы строк, пробелы, гигантскую длину) принимать нельзя.
    """
    candidate = (raw or "").strip()
    if candidate and _VALID_REQUEST_ID.match(candidate):
        return candidate
    return uuid.uuid4().hex


def current_request_id() -> str | None:
    """Request id текущего запроса (``None`` вне запроса)."""
    return _request_id.get()


def bind_request_id(value: str):
    """Привязывает id к контексту; возвращает токен для ``reset_request_id``."""
    return _request_id.set(value)


def reset_request_id(token) -> None:
    """Снимает привязку id (парная операция к ``bind_request_id``)."""
    _request_id.reset(token)


class RequestIdFilter(logging.Filter):
    """Добавляет ``record.request_id`` — поле для JSON и форматов с ``%(request_id)s``."""

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "request_id"):
            record.request_id = current_request_id() or "-"
        return True


class JsonFormatter(logging.Formatter):
    """Одна JSON-строка на запись: удобно для Loki/ELK и ``journalctl -o cat``."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": getattr(record, "request_id", "-"),
        }
        for key in ("method", "path", "status", "duration_ms"):
            value = getattr(record, key, None)
            if value is not None:
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def log_format() -> str:
    """Формат логов из ``DECISION_LOG_FORMAT``; неизвестный — ошибка."""
    raw = os.environ.get("DECISION_LOG_FORMAT", "").strip().lower()
    if raw in ("", "text"):
        return "text"
    if raw == "json":
        return "json"
    raise RuntimeError(
        f"DECISION_LOG_FORMAT must be one of {LOG_FORMATS} or empty, got {raw!r}"
    )


def log_level() -> int:
    """Уровень логов из ``DECISION_LOG_LEVEL`` (по умолчанию INFO)."""
    raw = os.environ.get("DECISION_LOG_LEVEL", "INFO").strip().upper()
    try:
        return int(logging.getLevelName(raw))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"DECISION_LOG_LEVEL must be a level name, got {raw!r}"
        ) from exc


_configured = False


def configure_logging() -> None:
    """Ставит JSON-формат и фильтр request id, если это попросили окружением.

    По умолчанию ничего не меняем: uvicorn и pytest настраивают логи сами, и
    лишний handler ломал бы ``caplog`` и дублировал строки. При
    ``DECISION_LOG_FORMAT=json`` добавляем ровно один handler на root.
    """
    global _configured
    if _configured:
        return
    _configured = True
    level = log_level()
    fmt = log_format()
    root = logging.getLogger()
    # Фильтр вешаем на все handlers (включая uvicorn), чтобы request id был
    # доступен в любом формате, где он упомянут.
    for handler in root.handlers:
        handler.addFilter(RequestIdFilter())
    if fmt != "json":
        # Без явного DECISION_LOG_LEVEL не трогаем уровень: его задаёт
        # uvicorn/pytest, и менять его из библиотеки — сюрприз.
        if os.environ.get("DECISION_LOG_LEVEL"):
            root.setLevel(level)
        return
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    handler.addFilter(RequestIdFilter())
    handler.setLevel(level)
    root.handlers = [handler]
    root.setLevel(level)


def reset_logging() -> None:
    """Сбрасывает флаг конфигурации (для тестов)."""
    global _configured
    _configured = False


def access_record(
    logger: logging.Logger,
    *,
    method: str,
    path: str,
    status: int,
    duration_ms: float,
    request_id: str,
) -> None:
    """Пишет строку доступа с полями, которые понимает ``JsonFormatter``."""
    logger.info(
        "%s %s -> %s %.1fms request_id=%s",
        method,
        path,
        status,
        duration_ms,
        request_id,
        extra={
            "request_id": request_id,
            "method": method,
            "path": path,
            "status": status,
            "duration_ms": round(duration_ms, 3),
        },
    )
