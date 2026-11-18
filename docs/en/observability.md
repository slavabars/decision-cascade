# Observability: metrics, request id and logs

**English** · [Русский](../observability.md)

The service exposes Prometheus metrics, sets `X-Request-ID` on every response and can write structured JSON logs. Request bodies (`state`, images, feedback examples) never reach metrics or logs — this is pinned by `tests/test_logging_privacy.py`.

## 1. `GET /metrics`

Prometheus text format 0.0.4, without authentication — as is customary for sidecar scraping. The endpoint is not counted in its own counters, so the scraper does not inflate `decision_http_requests_total`.

```bash
curl -fsS http://127.0.0.1:8080/metrics | head -20
```

| Metric | Type | Labels | Meaning |
| --- | --- | --- | --- |
| `decision_http_requests_total` | counter | `method`, `path`, `status` | HTTP requests by route template |
| `decision_http_request_duration_seconds` | histogram | `method`, `path` | request latency |
| `decision_decisions_total` | counter | `question`, `type`, `trained` | decisions returned |
| `decision_untrained_refusals_total` | counter | — | `409 UNTRAINED_HEADS` refusals |
| `decision_ready` | gauge | — | `1` — `/ready` answers `200`, `0` — `503` |
| `decision_heads_loaded` | gauge | — | how many trained packs are in the cache |

`path` is the route **template** (`/decisions`, `/v1/systemone`), not the actual URL: otherwise cardinality would grow with every request. Question names come from the request schema — their finite number is set by the client.

Standard process metrics (`python_gc_*`, `process_resident_memory_bytes`, etc.) are added by `prometheus_client` itself; they are useful when investigating leaks and memory degradation.

Metric names in the alert examples are checked against the real registry by `tests/test_docs.py::test_observability_alert_metrics_exist_in_registry`, and a live scrape is checked manually (§5): `decision_ready`, `decision_untrained_refusals_total` and `decision_heads_loaded` are returned with the expected values, the histogram with buckets.

### Scraping and alerts

```yaml
# prometheus.yml
scrape_configs:
  - job_name: decision-cascade
    scrape_interval: 15s
    static_configs:
      - targets: ["127.0.0.1:8080"]
```

Reasonable starting alerts:

```yaml
groups:
  - name: decision-cascade
    rules:
      - alert: DecisionNotReady
        expr: decision_ready == 0
        for: 5m
        annotations:
          summary: "decision-cascade is not ready to take traffic (/ready 503)"

      - alert: DecisionUntrainedRefusals
        expr: rate(decision_untrained_refusals_total[5m]) > 0
        for: 10m
        annotations:
          summary: "requests return 409: there is no trained pack for the question schema"

      - alert: DecisionLatencyHigh
        expr: |
          histogram_quantile(
            0.95,
            sum by (le, path) (rate(decision_http_request_duration_seconds_bucket[5m]))
          ) > 0.5
        for: 10m
        annotations:
          summary: "p95 latency is above 500 ms (the budget from docs/benchmarks.md)"
```

## 2. `X-Request-ID`

- If the client sends a valid `X-Request-ID` (printable characters, no spaces, up to 128 characters), it is preserved — convenient for correlating logs across services.
- Otherwise a new `uuid4().hex` is generated.
- The header is returned on **every** response, including errors.
- The value is available in logs through a `ContextVar`, so it reaches both the access log and `decision.*` messages inside the handler.

```bash
curl -i -H 'X-Request-ID: trace-42' -X POST http://127.0.0.1:8080/decisions \
  -H 'content-type: application/json' \
  -d '{"state":"Stripe failing","questions":{"urgent":{"type":"noul","instructions":"urgent?"}}}'
# X-Request-ID: trace-42
```

We do not pass arbitrary strings into the header: newlines and spaces are a log injection, so such a value is replaced with a new id.

## 3. Logs

By default the format is unchanged (Python/uvicorn) and the service reconfigures nothing. Structured logs are enabled by the environment:

| Variable | Values | Default |
| --- | --- | --- |
| `DECISION_LOG_FORMAT` | `text`, `json`, empty | empty (we do not interfere) |
| `DECISION_LOG_LEVEL` | level name (`DEBUG`, `INFO`, `WARNING`, …) | `INFO` |

```bash
DECISION_LOG_FORMAT=json DECISION_LOG_LEVEL=INFO \
  .venv/bin/python -m uvicorn api.server.main:app --port 8000
```

One line per request (`decision.access`):

```json
{"ts": "2026-10-29T12:00:00+0000", "level": "INFO", "logger": "decision.access",
 "message": "POST /decisions -> 200 12.3ms request_id=trace-42",
 "request_id": "trace-42", "method": "POST", "path": "/decisions",
 "status": 200, "duration_ms": 12.3}
```

The `method`/`path`/`status`/`duration_ms` fields are put into the JSON separately, so they can be filtered in Loki/ELK without parsing `message`. With `DECISION_LOG_FORMAT=json` the service installs exactly one handler on root; with `text`/empty it only adds a `request_id` filter without changing uvicorn and pytest handlers.

## 4. What is not logged

- `state` and prompts (the full state text);
- `images`/`videos` and media paths;
- `/v1/feedback` bodies (decisions and labels);
- authorization headers.

Decision logs keep the question name, type, distribution and `confidence` — enough to debug calibration, not enough to reconstruct the request text. Checks: `tests/test_logging_privacy.py`, `tests/test_observability.py`.

## 5. Manual check

```bash
# metrics grow after requests
curl -fsS http://127.0.0.1:8080/metrics | grep decision_http_requests_total

# the request id is returned and the client one is preserved
curl -isS -H 'X-Request-ID: trace-42' http://127.0.0.1:8080/health | grep -i x-request-id

# a single-line JSON log
DECISION_LOG_FORMAT=json .venv/bin/python -m uvicorn api.server.main:app --port 8000 \
  && curl -fsS http://127.0.0.1:8080/health
```
