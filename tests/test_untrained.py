"""Политика обслуживания необученных голов: allow / refuse / fallback и ``/info``."""
import pytest
from fastapi.testclient import TestClient

from api.server.service import UNTRAINED_POLICIES, DecisionService
from src.backends.base import BackendResult, Capabilities
from src.backends.cascade import CascadeBackend
from src.backends.local import LocalHeadsBackend
from src.errors import UntrainedDecisionError
from src.sampler.questions import NoulSpec
from src.schemas import NoulResult


def _questions() -> dict:
    return {"u": NoulSpec("urgent?")}


def test_allow_policy_serves_and_reports_untrained():
    service = DecisionService(dim=8)  # checkpoint_dir=None -> библиотечный allow
    backend = LocalHeadsBackend(service)

    result = backend.decide("state", _questions())

    assert "u" in result.decisions
    assert result.usage["trained"] is False
    assert service.is_trained(_questions()) is False


def test_refuse_policy_raises_for_unknown_schema(tmp_path):
    service = DecisionService(
        dim=8, checkpoint_dir=tmp_path, untrained_policy="refuse"
    )
    with pytest.raises(UntrainedDecisionError):
        service.decide("state", _questions())


def test_trained_heads_are_served_under_refuse_policy(tmp_path):
    questions = _questions()
    saver = DecisionService(dim=8, checkpoint_dir=tmp_path)
    saver.save_heads(8, questions)

    service = DecisionService(
        dim=8, checkpoint_dir=tmp_path, untrained_policy="refuse"
    )
    result = service.decide("state", questions)

    assert "u" in result
    assert service.is_trained(questions) is True


def test_invalid_policy_rejected():
    with pytest.raises(ValueError, match="untrained_policy"):
        DecisionService(dim=8, untrained_policy="whatever")


def test_info_reports_policy_and_heads(tmp_path):
    service = DecisionService(
        dim=8, checkpoint_dir=tmp_path, untrained_policy="refuse"
    )
    info = service.info()

    assert info["untrained_policy"] in UNTRAINED_POLICIES
    assert info["checkpoint_dir"] == str(tmp_path)
    assert info["dim"] == 8


def test_resolve_untrained_policy_auto_follows_checkpoint_env(monkeypatch):
    from api.server import main

    monkeypatch.delenv("DECISION_CHECKPOINT_DIR", raising=False)
    monkeypatch.delenv("DECISION_UNTRAINED", raising=False)
    assert main.resolve_untrained_policy() == "allow"

    monkeypatch.setenv("DECISION_CHECKPOINT_DIR", "checkpoints")
    assert main.resolve_untrained_policy() == "refuse"

    monkeypatch.setenv("DECISION_UNTRAINED", "fallback")
    assert main.resolve_untrained_policy() == "fallback"

    monkeypatch.setenv("DECISION_UNTRAINED", "nonsense")
    with pytest.raises(RuntimeError, match="DECISION_UNTRAINED"):
        main.resolve_untrained_policy()


# --- API -------------------------------------------------------------------

@pytest.fixture(scope="module")
def client():
    from api.server.main import app

    with TestClient(app) as c:
        yield c


PAYLOAD = {"state": "s", "questions": {"u": {"type": "noul", "instructions": "?"}}}


def test_api_reports_trained_flag(client, monkeypatch):
    from api.server import main

    backend = LocalHeadsBackend(DecisionService(dim=8))
    monkeypatch.setattr(main, "backend", backend)

    body = client.post("/decisions", json=PAYLOAD).json()

    assert body["trained"] is False


def test_api_refuses_untrained_with_409(client, monkeypatch, tmp_path):
    from api.server import main

    service = DecisionService(
        dim=8, checkpoint_dir=tmp_path, untrained_policy="refuse"
    )
    monkeypatch.setattr(main, "backend", LocalHeadsBackend(service))

    resp = client.post("/decisions", json=PAYLOAD)

    assert resp.status_code == 409
    assert resp.json()["error"]["code"] == "UNTRAINED_HEADS"


def test_api_info_endpoint(client):
    body = client.get("/info").json()

    assert body["backend"]
    assert "text" in body["modalities"]


# --- Каскад ----------------------------------------------------------------

class _StubBackend:
    def __init__(self, name: str):
        self.name = name
        self.capabilities = Capabilities()
        self.calls: list[tuple] = []

    def decide(self, state, questions, *, images=None, videos=None, media_kwargs=None):
        self.calls.append((state, tuple(questions)))
        return BackendResult(
            decisions={
                name: NoulResult(noul=0.9, confidence=0.9) for name in questions
            },
            usage={"trained": True},
        )


def test_cascade_falls_back_when_untrained(tmp_path):
    service = DecisionService(
        dim=8, checkpoint_dir=tmp_path, untrained_policy="fallback"
    )
    fallback = _StubBackend("remote")
    cascade = CascadeBackend(LocalHeadsBackend(service), fallback)

    result = cascade.decide("state", _questions())

    assert fallback.calls == [("state", ("u",))]
    assert result.decisions["u"].confidence == 0.9
    assert result.usage["reason"] == "untrained"


def test_cascade_propagates_refusal(tmp_path):
    service = DecisionService(
        dim=8, checkpoint_dir=tmp_path, untrained_policy="refuse"
    )
    cascade = CascadeBackend(LocalHeadsBackend(service), _StubBackend("remote"))

    with pytest.raises(UntrainedDecisionError):
        cascade.decide("state", _questions())
