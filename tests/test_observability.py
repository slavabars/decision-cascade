"""Наблюдаемость: request id, строка доступа и JSON-логи."""
from __future__ import annotations

import json
import logging

import pytest
from fastapi.testclient import TestClient

from api.server import observability


@pytest.fixture(scope="module")
def client():
    from api.server.main import app

    with TestClient(app) as c:
        yield c


@pytest.fixture
def clean_root_logging():
    root = logging.getLogger()
    saved = (list(root.handlers), root.level)
    observability.reset_logging()
    yield
    root.handlers = saved[0]
    root.setLevel(saved[1])
    observability.reset_logging()


def test_response_gets_generated_request_id(client):
    response = client.get("/health")
    request_id = response.headers[observability.REQUEST_ID_HEADER]
    assert request_id
    assert len(request_id) == 32  # uuid4().hex


def test_client_request_id_is_echoed(client):
    response = client.get("/health", headers={"X-Request-ID": "caller-123"})
    assert response.headers["X-Request-ID"] == "caller-123"


@pytest.mark.parametrize(
    "bad",
    [
        "has space",
        "line\nbreak",
        "x" * 200,
    ],
)
def test_invalid_client_request_id_is_replaced(client, bad):
    response = client.get("/health", headers={"X-Request-ID": bad})
    request_id = response.headers["X-Request-ID"]
    assert request_id != bad
    assert observability._VALID_REQUEST_ID.match(request_id)


def test_access_log_has_fields_and_no_body(client, caplog):
    with caplog.at_level(logging.INFO, logger="decision.access"):
        response = client.post(
            "/decisions",
            json={
                "state": "SECRET customer state",
                "questions": {"urgent": {"type": "noul", "instructions": "?"}},
            },
        )
    record = next(r for r in caplog.records if r.name == "decision.access")
    assert record.method == "POST"
    assert record.path == "/decisions"
    assert record.status == response.status_code
    assert record.duration_ms >= 0
    assert record.request_id == response.headers["X-Request-ID"]
    # тело запроса в лог не попадает
    assert "SECRET" not in record.getMessage()


def test_request_id_available_inside_handler(client):
    response = client.get("/health", headers={"X-Request-ID": "trace-1"})
    assert response.headers["X-Request-ID"] == "trace-1"


def test_new_request_id_rejects_unsafe_values():
    assert observability.new_request_id(None) != observability.new_request_id(None)
    assert observability.new_request_id("  abc.DEF_1-2  ") == "abc.DEF_1-2"


def test_bind_and_reset_request_id():
    assert observability.current_request_id() is None
    token = observability.bind_request_id("abc")
    try:
        assert observability.current_request_id() == "abc"
    finally:
        observability.reset_request_id(token)
    assert observability.current_request_id() is None


def test_json_formatter_emits_one_line_with_request_id():
    record = logging.LogRecord(
        name="decision.test",
        level=logging.WARNING,
        pathname=__file__,
        lineno=1,
        msg="hello %s",
        args=("world",),
        exc_info=None,
    )
    record.request_id = "rid-1"
    record.method = "GET"
    record.path = "/info"
    record.status = 200
    record.duration_ms = 1.5
    payload = json.loads(observability.JsonFormatter().format(record))
    assert payload["level"] == "WARNING"
    assert payload["message"] == "hello world"
    assert payload["request_id"] == "rid-1"
    assert payload["method"] == "GET"
    assert payload["path"] == "/info"
    assert payload["status"] == 200
    assert payload["duration_ms"] == 1.5
    assert "\n" not in json.dumps(payload)


def test_request_id_filter_fills_missing_field():
    record = logging.LogRecord(
        name="decision.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="x",
        args=(),
        exc_info=None,
    )
    token = observability.bind_request_id("rid-2")
    try:
        assert observability.RequestIdFilter().filter(record) is True
    finally:
        observability.reset_request_id(token)
    assert record.request_id == "rid-2"


def test_log_level_and_format_parsing(monkeypatch):
    monkeypatch.setenv("DECISION_LOG_LEVEL", "debug")
    assert observability.log_level() == logging.DEBUG
    monkeypatch.setenv("DECISION_LOG_LEVEL", "not-a-level")
    with pytest.raises(ValueError):
        observability.log_level()

    monkeypatch.delenv("DECISION_LOG_FORMAT", raising=False)
    assert observability.log_format() == "text"
    monkeypatch.setenv("DECISION_LOG_FORMAT", "JSON")
    assert observability.log_format() == "json"
    monkeypatch.setenv("DECISION_LOG_FORMAT", "xml")
    with pytest.raises(RuntimeError):
        observability.log_format()


def test_configure_logging_json_installs_single_handler(
    monkeypatch, clean_root_logging
):
    monkeypatch.setenv("DECISION_LOG_FORMAT", "json")
    monkeypatch.setenv("DECISION_LOG_LEVEL", "WARNING")
    observability.configure_logging()

    root = logging.getLogger()
    assert len(root.handlers) == 1
    handler = root.handlers[0]
    assert isinstance(handler.formatter, observability.JsonFormatter)
    assert root.level == logging.WARNING
    assert any(
        isinstance(flt, observability.RequestIdFilter) for flt in handler.filters
    )

    # повторный вызов не плодит handlers
    observability.configure_logging()
    assert len(root.handlers) == 1


def test_configure_logging_text_keeps_handlers(monkeypatch, clean_root_logging):
    root = logging.getLogger()
    sentinel = logging.NullHandler()
    root.handlers = [sentinel]
    monkeypatch.delenv("DECISION_LOG_FORMAT", raising=False)
    observability.configure_logging()
    assert root.handlers == [sentinel]
    assert any(isinstance(flt, observability.RequestIdFilter) for flt in sentinel.filters)
