"""Liveness/readiness и типизированные ошибки внешнего бэкенда."""
import httpx
import pytest
from fastapi.testclient import TestClient

from src.backends.base import BackendResult, Capabilities
from src.backends.cascade import CascadeBackend
from src.backends.http import HttpSystemOneBackend
from src.backends.local import LocalHeadsBackend
from src.errors import RemoteBackendError
from src.sampler.questions import NoulSpec


@pytest.fixture(scope="module")
def client():
    from api.server.main import app

    with TestClient(app) as c:
        yield c


PAYLOAD = {
    "state": "Stripe failing for 3 days",
    "questions": {"urgent": {"type": "noul", "instructions": "urgent?"}},
}


def test_health_is_alive_even_when_backend_is_not(client, monkeypatch):
    from api.server import main

    class _Broken:
        name = "broken"
        capabilities = Capabilities()

        def readiness(self):
            raise RuntimeError("boom")

    monkeypatch.setattr(main, "backend", _Broken())
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
    assert resp.json()["version"]


def test_ready_reports_local_service(client):
    resp = client.get("/ready")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ready"] is True
    assert body["backend"] == "local"
    assert body["encoder_loaded"] is True


def test_ready_is_503_when_refuse_policy_has_no_checkpoints(client, monkeypatch, tmp_path):
    from api.server import main
    from api.server.service import DecisionService

    service = DecisionService(
        dim=8, checkpoint_dir=tmp_path, untrained_policy="refuse"
    )
    monkeypatch.setattr(main, "backend", LocalHeadsBackend(service))
    resp = client.get("/ready")
    assert resp.status_code == 503
    body = resp.json()
    assert body["ready"] is False
    assert any("no trained checkpoints" in reason for reason in body["reasons"])


def test_ready_is_503_when_encoder_is_not_loaded(client, monkeypatch):
    from api.server import main
    from api.server.service import DecisionService

    class _NotLoadedEncoder:
        _model = None
        dim = 8

        def encode(self, states):
            raise AssertionError("must not be called")

    service = DecisionService(encoder=_NotLoadedEncoder(), dim=8)
    monkeypatch.setattr(main, "backend", LocalHeadsBackend(service))
    resp = client.get("/ready")
    assert resp.status_code == 503
    assert "encoder is not loaded" in resp.json()["reasons"]


def test_warmup_loads_checkpoint_and_makes_service_ready(tmp_path):
    from api.server.service import DecisionService
    from src.decision_head.heads import DecisionHeads

    questions = {"urgent": NoulSpec("urgent?")}
    DecisionHeads(8, questions, seed=0).save_checkpoint(tmp_path / "pack.pt")
    service = DecisionService(
        dim=8, checkpoint_dir=tmp_path, untrained_policy="refuse"
    )

    assert service.warmup() == 1
    report = service.readiness()
    assert report["ready"] is True
    assert report["heads_loaded"] == 1
    assert report["checkpoints"] == 1


def test_ready_is_503_when_checkpoints_are_unreadable(tmp_path):
    from api.server.service import DecisionService

    (tmp_path / "pack.pt").write_bytes(b"not a torch checkpoint")
    service = DecisionService(
        dim=8, checkpoint_dir=tmp_path, untrained_policy="refuse"
    )

    assert service.warmup() == 0
    report = service.readiness()
    assert report["ready"] is False
    assert any("not readable" in reason for reason in report["reasons"])


def test_cascade_ready_when_fallback_is_down(client, monkeypatch):
    from api.server import main

    class _Primary:
        name = "local"
        capabilities = Capabilities()

        def readiness(self):
            return {"ready": True, "encoder_loaded": True}

    class _Fallback:
        name = "remote"
        capabilities = Capabilities()

        def readiness(self):
            return {"ready": False, "reason": "ConnectError"}

    monkeypatch.setattr(
        main, "backend", CascadeBackend(_Primary(), _Fallback())
    )
    resp = client.get("/ready")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ready"] is True
    assert body["degraded"] is True
    assert body["fallback"]["reason"] == "ConnectError"


def test_http_readiness_probes_health():
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json={}))
    backend = HttpSystemOneBackend("http://remote", transport=transport)
    report = backend.readiness()
    assert report["ready"] is True
    assert report["remote_status"] == 200


def test_http_readiness_marks_5xx_as_not_ready():
    transport = httpx.MockTransport(lambda request: httpx.Response(503, json={}))
    backend = HttpSystemOneBackend("http://remote", transport=transport)
    assert backend.readiness()["ready"] is False


def test_http_readiness_marks_transport_error_as_not_ready():
    def handler(request):
        raise httpx.ConnectError("boom", request=request)

    backend = HttpSystemOneBackend(
        "http://remote", transport=httpx.MockTransport(handler)
    )
    report = backend.readiness()
    assert report["ready"] is False
    assert report["reason"] == "ConnectError"


class _UnavailableBackend:
    name = "remote"
    capabilities = Capabilities()

    def decide(self, state, questions, **kwargs):
        raise RemoteBackendError("remote: HTTP 503", backend="remote", status_code=503)

    def decide_many(self, states, questions, **kwargs):
        raise RemoteBackendError("remote: HTTP 503", backend="remote", status_code=503)


def test_decisions_returns_502_when_remote_is_unavailable(client, monkeypatch):
    from api.server import main

    monkeypatch.setattr(main, "backend", _UnavailableBackend())
    resp = client.post("/decisions", json=PAYLOAD)
    assert resp.status_code == 502
    error = resp.json()["error"]
    assert error["code"] == "BACKEND_UNAVAILABLE"
    assert error["backend"] == "remote"
    assert error["status_code"] == 503
    assert "Stripe" not in resp.text


def test_batch_returns_502_when_remote_is_unavailable(client, monkeypatch):
    from api.server import main

    monkeypatch.setattr(main, "backend", _UnavailableBackend())
    resp = client.post(
        "/decisions/batch",
        json={"states": ["one"], "questions": PAYLOAD["questions"]},
    )
    assert resp.status_code == 502
    assert resp.json()["error"]["code"] == "BACKEND_UNAVAILABLE"


def test_systemone_returns_502_when_remote_is_unavailable(client, monkeypatch):
    from api.server import main

    monkeypatch.setattr(main, "backend", _UnavailableBackend())
    resp = client.post(
        "/v1/systemone",
        json={"state": "Stripe failing", "questions": PAYLOAD["questions"]},
    )
    assert resp.status_code == 502
    assert resp.json()["error"]["code"] == "BACKEND_UNAVAILABLE"
    assert "Stripe" not in resp.text


def test_remote_backend_error_does_not_carry_payload():
    error = RemoteBackendError("remote: HTTP 500", backend="remote", status_code=500)
    assert "state" not in str(error)
    assert error.status_code == 500


def test_http_backend_options_parse_env(monkeypatch):
    from api.server import main

    monkeypatch.setenv("DECISION_BACKEND_TIMEOUT", "12.5")
    monkeypatch.setenv("DECISION_BACKEND_RETRIES", "0")
    assert main.http_backend_options() == {"timeout": 12.5, "retries": 0}

    monkeypatch.setenv("DECISION_BACKEND_TIMEOUT", "0")
    with pytest.raises(RuntimeError, match="must be > 0"):
        main.http_backend_options()

    monkeypatch.setenv("DECISION_BACKEND_TIMEOUT", "10")
    monkeypatch.setenv("DECISION_BACKEND_RETRIES", "-1")
    with pytest.raises(RuntimeError, match="must be >= 0"):
        main.http_backend_options()


def test_local_backend_readiness_delegates():
    from api.server.service import DecisionService

    backend = LocalHeadsBackend(DecisionService(dim=8))
    report = backend.readiness()
    assert report["ready"] is True
    assert report["backend"] == "local"


def test_backend_without_readiness_is_ready():
    class _Legacy:
        name = "legacy"
        capabilities = Capabilities()

        def decide(self, state, questions, **kwargs):
            return BackendResult(decisions={}, usage={})

    from src.backends.base import backend_readiness

    assert backend_readiness(_Legacy()) == {"ready": True}
