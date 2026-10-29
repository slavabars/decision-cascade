"""Политика депрекации: реестр, заголовки ответа и предупреждение в лог."""
from __future__ import annotations

import logging

import pytest
from fastapi.testclient import TestClient

from api.server.deprecation import (
    DEPRECATIONS,
    MIN_MINOR_RELEASES,
    Deprecation,
    minor_distance,
    parse_minor,
    reset_warnings,
    validate_registry,
    warn_once,
)


@pytest.fixture(autouse=True)
def _clean_warnings():
    reset_warnings()
    yield
    reset_warnings()


@pytest.fixture(scope="module")
def client():
    from api.server.main import app

    with TestClient(app) as c:
        yield c


def test_parse_minor_accepts_semver_prefix():
    assert parse_minor("0.13.0") == (0, 13)
    assert parse_minor("1.2") == (1, 2)


@pytest.mark.parametrize("bad", ["", "1", "x.y", "1.x"])
def test_parse_minor_rejects_malformed(bad):
    with pytest.raises(ValueError):
        parse_minor(bad)


def test_minor_distance_counts_minor_releases():
    assert minor_distance("0.13.0", "0.15.0") == MIN_MINOR_RELEASES
    assert minor_distance("0.13.0", "0.14.0") == 1
    assert minor_distance("0.13.0", "1.0.0") > MIN_MINOR_RELEASES


def test_deprecation_rejects_short_window():
    with pytest.raises(ValueError, match="minimum is 2"):
        Deprecation(item="/old", since="0.13.0", remove_in="0.14.0")


def test_deprecation_headers():
    deprecation = Deprecation(
        item="/old",
        since="0.13.0",
        remove_in="0.15.0",
        alternative="/v1/old",
        sunset="Tue, 01 Sep 2026 00:00:00 GMT",
    )
    assert deprecation.headers() == {
        "Deprecation": "true",
        "Sunset": "Tue, 01 Sep 2026 00:00:00 GMT",
        "Link": '</v1/old>; rel="deprecation"',
    }


def test_headers_without_alternative_or_sunset():
    deprecation = Deprecation(item="DECISION_OLD", since="0.13.0", remove_in="0.15.0")
    assert deprecation.headers() == {"Deprecation": "true"}


def test_shipped_registry_is_valid():
    validate_registry()
    # Реестр пуст до первой реальной депрекации — это осознанное состояние.
    assert DEPRECATIONS == {}


def test_middleware_marks_deprecated_path_once(client, monkeypatch, caplog):
    deprecation = Deprecation(
        item="/info",
        since="0.13.0",
        remove_in="0.15.0",
        alternative="/v2/info",
    )
    monkeypatch.setitem(DEPRECATIONS, "/info", deprecation)
    with caplog.at_level(logging.WARNING, logger="decision.deprecation"):
        first = client.get("/info")
        second = client.get("/info")
    assert first.headers["Deprecation"] == "true"
    assert first.headers["Link"] == '</v2/info>; rel="deprecation"'
    assert second.headers["Deprecation"] == "true"
    messages = [r.message for r in caplog.records if r.name == "decision.deprecation"]
    assert len(messages) == 1
    assert "removal in 0.15.0" in messages[0]


def test_middleware_leaves_live_paths_untouched(client):
    response = client.get("/health")
    assert "Deprecation" not in response.headers
    assert "Sunset" not in response.headers


def test_warn_once_is_idempotent(monkeypatch, caplog):
    deprecation = Deprecation(item="DECISION_OLD", since="0.13.0", remove_in="0.15.0")
    monkeypatch.setitem(DEPRECATIONS, "DECISION_OLD", deprecation)
    with caplog.at_level(logging.WARNING, logger="decision.deprecation"):
        assert warn_once("DECISION_OLD") is deprecation
        assert warn_once("DECISION_OLD") is deprecation
    assert len(caplog.records) == 1


def test_warn_once_ignores_unknown_item(caplog):
    assert warn_once("/never-declared") is None
    assert caplog.records == []
