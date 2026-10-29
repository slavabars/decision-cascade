# Наблюдаемость: метрики, request id и логи

Сервис отдаёт метрики Prometheus, проставляет `X-Request-ID` на каждый ответ и
умеет писать структурированные JSON-логи. Тела запросов (`state`, изображения,
примеры обратной связи) ни в метрики, ни в логи не попадают — это закреплено
`tests/test_logging_privacy.py`.

## 1. `GET /metrics`

Формат Prometheus text 0.0.4, без авторизации — как принято для sidecar-скрейпа.
Эндпоинт не учитывается в собственных счётчиках, чтобы скрейпер не раздувал
`decision_http_requests_total`.

```bash
curl -fsS http://127.0.0.1:8080/metrics | head -20
```

| Метрика | Тип | Метки | Смысл |
| --- | --- | --- | --- |
| `decision_http_requests_total` | counter | `method`, `path`, `status` | HTTP-запросы по шаблону маршрута |
| `decision_http_request_duration_seconds` | histogram | `method`, `path` | латентность запроса |
| `decision_decisions_total` | counter | `question`, `type`, `trained` | выданные решения |
| `decision_untrained_refusals_total` | counter | — | отказы `409 UNTRAINED_HEADS` |
| `decision_ready` | gauge | — | `1` — `/ready` отвечает `200`, `0` — `503` |
| `decision_heads_loaded` | gauge | — | сколько обученных паков в кэше |

`path` — **шаблон** маршрута (`/decisions`, `/v1/systemone`), а не фактический
URL: иначе кардинальность росла бы с каждым запросом. Имена вопросов берутся из
схемы запроса — их конечное число задаёт клиент.

Стандартные метрики процесса (`python_gc_*`, `process_resident_memory_bytes` и
т.п.) добавляет сам `prometheus_client`; они полезны при разборе утечек и
деградации памяти.

### Скрейп и алерты

```yaml
# prometheus.yml
scrape_configs:
  - job_name: decision-cascade
    scrape_interval: 15s
    static_configs:
      - targets: ["127.0.0.1:8080"]
```

Разумные стартовые алерты:

```yaml
groups:
  - name: decision-cascade
    rules:
      - alert: DecisionNotReady
        expr: decision_ready == 0
        for: 5m
        annotations:
          summary: "decision-cascade не готов принимать трафик (/ready 503)"

      - alert: DecisionUntrainedRefusals
        expr: rate(decision_untrained_refusals_total[5m]) > 0
        for: 10m
        annotations:
          summary: "запросы уходят в 409: под схему вопросов нет обученного пака"

      - alert: DecisionLatencyHigh
        expr: |
          histogram_quantile(
            0.95,
            sum by (le, path) (rate(decision_http_request_duration_seconds_bucket[5m]))
          ) > 0.5
        for: 10m
        annotations:
          summary: "p95 латентности выше 500 мс (бюджет из docs/benchmarks.md)"
```

## 2. `X-Request-ID`

- Если клиент прислал валидный `X-Request-ID` (печатные символы, без пробелов,
  до 128 символов), он сохраняется — удобно связывать логи нескольких сервисов.
- Иначе генерируется новый `uuid4().hex`.
- Заголовок возвращается в ответе на **любой** запрос, включая ошибки.
- Значение доступно в логах через `ContextVar`, поэтому попадает и в лог
  доступа, и в сообщения `decision.*` внутри обработчика.

```bash
curl -i -H 'X-Request-ID: trace-42' -X POST http://127.0.0.1:8080/decisions \
  -H 'content-type: application/json' \
  -d '{"state":"Stripe failing","questions":{"urgent":{"type":"noul","instructions":"urgent?"}}}'
# X-Request-ID: trace-42
```

Произвольную строку в заголовок не пропускаем: переводы строк и пробелы — это
инъекция в лог, поэтому такое значение заменяется на новый id.

## 3. Логи

По умолчанию формат прежний (Python/uvicorn), сервис ничего не перенастраивает.
Структурированные логи включаются окружением:

| Переменная | Значения | По умолчанию |
| --- | --- | --- |
| `DECISION_LOG_FORMAT` | `text`, `json`, пусто | пусто (не вмешиваемся) |
| `DECISION_LOG_LEVEL` | имя уровня (`DEBUG`, `INFO`, `WARNING`, …) | `INFO` |

```bash
DECISION_LOG_FORMAT=json DECISION_LOG_LEVEL=INFO \
  .venv/bin/python -m uvicorn api.server.main:app --port 8000
```

Одна строка на запрос (`decision.access`):

```json
{"ts": "2026-10-29T12:00:00+0000", "level": "INFO", "logger": "decision.access",
 "message": "POST /decisions -> 200 12.3ms request_id=trace-42",
 "request_id": "trace-42", "method": "POST", "path": "/decisions",
 "status": 200, "duration_ms": 12.3}
```

Поля `method`/`path`/`status`/`duration_ms` попадают в JSON отдельно, поэтому по
ним можно фильтровать в Loki/ELK, не разбирая `message`. При
`DECISION_LOG_FORMAT=json` сервис ставит ровно один handler на root; при
`text`/пусто — только добавляет фильтр `request_id`, не меняя обработчики
uvicorn и pytest.

## 4. Что не логируется

- `state` и промпты (полный текст состояния);
- `images`/`videos` и пути к медиа;
- тела `/v1/feedback` (решения и метки);
- заголовки авторизации.

В логах решений остаются имя вопроса, тип, распределение и `confidence` — этого
достаточно для отладки калибровки и недостаточно для восстановления текста
запроса. Проверка: `tests/test_logging_privacy.py`, `tests/test_observability.py`.

## 5. Проверка вручную

```bash
# метрики растут после запросов
curl -fsS http://127.0.0.1:8080/metrics | grep decision_http_requests_total

# request id возвращается и сохраняется клиентский
curl -isS -H 'X-Request-ID: trace-42' http://127.0.0.1:8080/health | grep -i x-request-id

# JSON-лог одной строкой
DECISION_LOG_FORMAT=json .venv/bin/python -m uvicorn api.server.main:app --port 8000 \
  && curl -fsS http://127.0.0.1:8080/health
```
