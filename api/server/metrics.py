"""Prometheus-метрики сервиса: запросы, латентность, решения, готовность.

Метки намеренно низкокардинальные: путь берётся из шаблона маршрута
(``/decisions``), а не из фактического URL, имена вопросов — из схемы запроса.
Тело запроса (``state``, изображения) в метки не попадает — это тот же принцип,
что и в логах (``tests/test_logging_privacy.py``).

``GET /metrics`` отдаёт формат Prometheus 0.0.4. Метрики процесса (GC, память)
добавляет сам ``prometheus_client`` — они полезны при разборе утечек.
"""
from __future__ import annotations

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    REGISTRY,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

#: Пути, которые не считаем работой сервиса: их дёргают пробы и скрейпер.
UNMETERED_PATHS = frozenset({"/metrics"})

REQUESTS = Counter(
    "decision_http_requests_total",
    "HTTP-запросы по методу, шаблону пути и коду ответа.",
    ("method", "path", "status"),
)

LATENCY = Histogram(
    "decision_http_request_duration_seconds",
    "Длительность HTTP-запроса в секундах.",
    ("method", "path"),
    # от быстрых /health до внешнего вызова с бюджетом в секунды
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)

DECISIONS = Counter(
    "decision_decisions_total",
    "Выданные решения по вопросу, типу и статусу обученности.",
    ("question", "type", "trained"),
)

UNTRAINED_REFUSALS = Counter(
    "decision_untrained_refusals_total",
    "Отказы 409 UNTRAINED_HEADS: под схему вопросов нет обученного пака.",
)

READY = Gauge(
    "decision_ready",
    "1, если /ready отвечает 200; 0 — сервис не готов принимать трафик.",
)

HEADS_LOADED = Gauge(
    "decision_heads_loaded",
    "Сколько обученных паков загружено в кэш сервиса.",
)


def route_template(request) -> str:
    """Шаблон пути для метки: ``/decisions`` вместо произвольного URL.

    FastAPI кладёт найденный маршрут в ``scope["route"]``; для 404 его нет —
    тогда используется сам путь (он и так ограничен набором известных).
    """
    route = request.scope.get("route")
    template = getattr(route, "path", None)
    if template:
        return template
    return request.url.path


def observe_request(
    method: str, path: str, status: int, seconds: float
) -> None:
    """Учитывает один HTTP-запрос, если путь вообще подлежит учёту."""
    if path in UNMETERED_PATHS:
        return
    REQUESTS.labels(method=method, path=path, status=str(status)).inc()
    LATENCY.labels(method=method, path=path).observe(seconds)


def observe_decisions(
    decisions: dict[str, object], trained: bool | None
) -> None:
    """Считает решения по вопросам; ``trained`` — статус из ответа."""
    label = "unknown" if trained is None else ("true" if trained else "false")
    for name, decision in decisions.items():
        DECISIONS.labels(
            question=name,
            type=str(getattr(decision, "type", "unknown")),
            trained=label,
        ).inc()


def observe_refusal() -> None:
    """Считает отказ ``UNTRAINED_HEADS`` (409)."""
    UNTRAINED_REFUSALS.inc()


def set_ready(ready: bool) -> None:
    """Обновляет gauge готовности по фактическому ответу ``/ready``."""
    READY.set(1 if ready else 0)


def set_heads_loaded(count: int) -> None:
    """Обновляет gauge числа загруженных паков."""
    HEADS_LOADED.set(count)


def render() -> bytes:
    """Сериализует реестр метрик в формат Prometheus."""
    return generate_latest(REGISTRY)


__all__ = [
    "CONTENT_TYPE_LATEST",
    "observe_decisions",
    "observe_refusal",
    "observe_request",
    "render",
    "route_template",
    "set_heads_loaded",
    "set_ready",
]
