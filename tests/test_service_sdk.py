"""Тесты SDK и сервиса — написаны до реализации (TDD RED)."""
import httpx
import pytest

from api.server.service import DecisionService, HashStateEncoder
from src.sampler.questions import ChoiceSpec, NoulSpec, ScoreSpec
from src.schemas import SCHEMA_VERSION


def test_hash_encoder_deterministic():
    enc = HashStateEncoder(dim=16)
    a = enc.encode(["stripe failing"])
    b = enc.encode(["stripe failing"])
    assert (a == b).all()
    assert a.shape == (1, 16)
    assert abs(a.norm().item() - 1.0) < 1e-6


def test_hash_encoder_different_states_differ():
    enc = HashStateEncoder(dim=16)
    a = enc.encode(["stripe failing"])
    b = enc.encode(["weather and gifts"])
    assert not (a == b).all()


def test_service_returns_typed_results():
    service = DecisionService(seed=1)
    questions = {
        "u": NoulSpec(instructions="?"),
        "c": ChoiceSpec(instructions="?", options=("a", "b")),
        "s": ScoreSpec(instructions="?", levels=("low", "high")),
    }
    results = service.decide("stripe failing for days", questions)
    assert set(results) == {"u", "c", "s"}
    assert results["u"].type == "noul"
    assert results["c"].type == "choice"
    assert results["s"].type == "score"


def test_service_seed_reproducible():
    q = {"u": NoulSpec(instructions="?")}
    r1 = DecisionService(seed=7).decide("same state", q)["u"]
    r2 = DecisionService(seed=7).decide("same state", q)["u"]
    assert r1.noul == r2.noul


def test_service_injectable_encoder():
    class Fixed:
        def encode(self, states):
            import torch

            return torch.ones(len(states), 4)

    svc = DecisionService(encoder=Fixed(), dim=4)
    out = svc.decide("anything", {"u": NoulSpec(instructions="?")})["u"]
    assert out.type == "noul"


def _json_payload():
    return {
        "schema_version": SCHEMA_VERSION,
        "decisions": {
            "u": {"type": "noul", "noul": 0.9, "confidence": 0.8},
            "c": {
                "type": "choice",
                "choice": "x",
                "probabilities": {"x": 0.6, "y": 0.4},
                "confidence": 0.6,
            },
        },
    }


def test_sdk_decisions_request_and_validate():
    from api.sdk.client import DecisionClient, validate_response

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/decisions"
        return httpx.Response(200, json=_json_payload())

    client = DecisionClient(transport=httpx.MockTransport(handler))
    assert validate_response(_json_payload()) is None
    result = client.decisions("state", {"u": {"type": "noul", "instructions": "?"}})
    assert result["decisions"]["u"]["noul"] == 0.9
    client.close()


def test_sdk_validation_rejects_bad_distribution():
    from api.sdk.client import validate_response

    bad = _json_payload()
    bad["decisions"]["c"]["probabilities"] = {"x": 0.9, "y": 0.9}
    with pytest.raises(ValueError):
        validate_response(bad)


def test_sdk_validation_rejects_missing_confidence():
    from api.sdk.client import validate_response

    bad = _json_payload()
    del bad["decisions"]["u"]["confidence"]
    with pytest.raises(ValueError):
        validate_response(bad)