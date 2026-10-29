"""Тесты абстракции бэкендов: локальные головы и внешний SystemOne-сервис."""
import json

import httpx
import pytest

from src.backends.base import BackendResult, Capabilities
from src.backends.http import HttpSystemOneBackend, parse_modalities
from src.backends.local import LocalHeadsBackend
from src.backends.protocol import answer_to_result, result_to_answer, spec_to_request
from src.sampler.questions import ChoiceSpec, NoulSpec, ScoreSpec
from src.schemas import ChoiceResult, NoulResult, ScoreResult


def _local_backend():
    from api.server.service import DecisionService

    return LocalHeadsBackend(DecisionService(seed=0))


def test_capabilities_default_text_only():
    caps = Capabilities()
    assert caps.modalities == frozenset({"text"})
    assert "image" not in caps.modalities


def test_local_backend_returns_backend_result():
    backend = _local_backend()
    assert backend.name
    assert backend.capabilities.modalities == frozenset({"text"})
    result = backend.decide("some state", {"u": NoulSpec("urgent?")})
    assert isinstance(result, BackendResult)
    assert result.decisions["u"].type == "noul"
    assert result.usage["questions"] == 1


def test_local_backend_capabilities_reflect_vision():
    from api.server.service import DecisionService
    from src.encoder.vision import HashVisionEncoder

    service = DecisionService(seed=0, vision_encoder=HashVisionEncoder(dim=64))
    backend = LocalHeadsBackend(service)
    assert backend.capabilities.modalities == frozenset(
        {"text", "image", "video"}
    )


def test_local_backend_forwards_media_to_engine():
    class SpyEngine:
        def __init__(self):
            self.seen: tuple | None = None

        @property
        def modalities(self):
            return frozenset({"text", "image"})

        def decide(
            self, state, questions, *, images=None, videos=None, media_kwargs=None
        ):
            self.seen = (images, videos, media_kwargs)
            return {}

    engine = SpyEngine()
    backend = LocalHeadsBackend(engine)
    assert backend.capabilities.modalities == frozenset({"text", "image"})
    backend.decide("s", {}, images=["i"], media_kwargs={"a": 1})
    assert engine.seen == (["i"], None, {"a": 1})


def test_local_backend_handles_all_types():
    backend = _local_backend()
    questions = {
        "u": NoulSpec("?"),
        "c": ChoiceSpec("?", ("a", "b")),
        "s": ScoreSpec("?", ("low", "high")),
    }
    result = backend.decide("state", questions)
    assert {n: r.type for n, r in result.decisions.items()} == {
        "u": "noul",
        "c": "choice",
        "s": "score",
    }


def _systemone_response():
    return {
        "model": "remote",
        "answers": {
            "u": {"type": "noul", "probability": 0.7, "confidence": 0.4},
            "c": {
                "type": "choice",
                "choice": "a",
                "confidence": 0.6,
                "probabilities": {"a": 0.6, "b": 0.4},
            },
            "s": {
                "type": "score",
                "score": 1.0,
                "confidence": 0.5,
                "probabilities": {"low": 0.5, "high": 0.5},
                "legend": ["low", "high"],
            },
        },
        "usage": {"questions": 3},
    }


def test_http_backend_forwards_and_parses():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/v1/systemone"
        payload = json.loads(request.content)
        assert payload["state"] == "state text"
        assert payload["questions"]["u"]["type"] == "noul"
        return httpx.Response(200, json=_systemone_response())

    backend = HttpSystemOneBackend(
        "http://remote",
        model="remote",
        modalities=("text", "image"),
        transport=httpx.MockTransport(handler),
    )
    result = backend.decide(
        "state text",
        {
            "u": NoulSpec("urgent?"),
            "c": ChoiceSpec("who?", ("a", "b")),
            "s": ScoreSpec("how bad?", ("low", "high")),
        },
    )
    assert result.decisions["u"].noul == 0.7
    assert result.decisions["c"].choice == "a"
    assert result.decisions["s"].legend == ("low", "high")
    assert result.usage == {"questions": 3}


def test_http_backend_passes_media_through():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(200, json={"model": "r", "answers": {}, "usage": {}})

    backend = HttpSystemOneBackend(
        "http://remote",
        modalities=("text", "image", "video"),
        transport=httpx.MockTransport(handler),
    )
    backend.decide("s", {"q": NoulSpec("?")}, images=["img"], videos=["vid"])
    assert captured["images"] == ["img"]
    assert captured["videos"] == ["vid"]


def test_result_to_answer_roundtrip_all_types():
    results = [
        NoulResult(noul=0.8, confidence=0.6),
        ChoiceResult(
            choice="a", probabilities={"a": 0.7, "b": 0.3}, confidence=0.7
        ),
        ScoreResult(
            score=1.0,
            probabilities={"low": 0.5, "high": 0.5},
            confidence=0.5,
            legend=("low", "high"),
        ),
    ]
    for result in results:
        assert answer_to_result(result_to_answer(result)) == result


def test_answer_to_result_derives_confidence_for_noul():
    parsed = answer_to_result({"type": "noul", "probability": 0.9})
    assert parsed.confidence == pytest.approx(0.8)


def test_spec_to_request_omits_empty_criteria():
    assert "criteria" not in spec_to_request(NoulSpec("?", None))
    assert spec_to_request(ChoiceSpec("?", ("a", "b")))["options"] == ["a", "b"]


def test_parse_modalities_valid_and_invalid():
    assert parse_modalities("text, image") == ("text", "image")
    assert parse_modalities("") == ("text",)
    with pytest.raises(ValueError):
        parse_modalities("text,audio")


def test_http_backend_context_manager_closes():
    backend = HttpSystemOneBackend("http://remote")
    with backend as entered:
        assert entered is backend
    assert backend._client.is_closed


def test_local_backend_decide_many_uses_engine_batch():
    class BatchEngine:
        def __init__(self):
            self.many_calls = 0

        @property
        def modalities(self):
            return frozenset({"text"})

        def decide(self, *args, **kwargs):  # pragma: no cover - must not be called
            raise AssertionError("single-state decide should not be used for a text batch")

        def decide_many(self, states, questions):
            self.many_calls += 1
            return [
                {name: NoulResult(noul=0.5, confidence=0.0) for name in questions}
                for _ in states
            ]

    engine = BatchEngine()
    backend = LocalHeadsBackend(engine)
    results = backend.decide_many(["a", "b"], {"u": NoulSpec("?")})
    assert engine.many_calls == 1
    assert [set(r.decisions) for r in results] == [{"u"}, {"u"}]
    assert all(r.usage["questions"] == 1 for r in results)


def test_create_backend_defaults_to_local(monkeypatch):
    from api.server import main

    monkeypatch.delenv("DECISION_BACKEND", raising=False)
    backend = main.create_backend()
    assert isinstance(backend, LocalHeadsBackend)


def test_create_backend_selects_http(monkeypatch):
    from api.server import main

    monkeypatch.setenv("DECISION_BACKEND", "http")
    monkeypatch.setenv("DECISION_BACKEND_URL", "http://localhost:30000")
    monkeypatch.setenv("DECISION_BACKEND_MODALITIES", "text,image,video")
    backend = main.create_backend()
    assert isinstance(backend, HttpSystemOneBackend)
    assert "video" in backend.capabilities.modalities


def test_answer_to_result_normalizes_rounded_probabilities():
    from src.backends.protocol import answer_to_result

    choice = answer_to_result(
        {
            "type": "choice",
            "choice": "Business",
            "probabilities": {
                "World": 0.0016,
                "Sports": 0.0003,
                "Business": 0.8241,
                "Sci/Tech": 0.1741,
            },
            "confidence": 0.8241,
        }
    )
    assert sum(choice.probabilities.values()) == pytest.approx(1.0)
    assert choice.probabilities["Business"] == pytest.approx(0.8241, abs=1e-3)


def test_answer_to_result_normalizes_score_probabilities():
    from src.backends.protocol import answer_to_result

    score = answer_to_result(
        {
            "type": "score",
            "score": 1.0,
            "probabilities": {"low": 0.0016, "medium": 0.8241, "high": 0.1741},
            "confidence": 0.8241,
            "legend": ["low", "medium", "high"],
        }
    )
    assert sum(score.probabilities.values()) == pytest.approx(1.0)


def _retry_payload():
    return {"u": NoulSpec("urgent?")}


def test_http_backend_retries_transient_status():
    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(503, json={"error": "busy"})
        return httpx.Response(200, json=_systemone_response())

    backend = HttpSystemOneBackend(
        "http://remote", retries=1, transport=httpx.MockTransport(handler)
    )
    result = backend.decide("state text", _retry_payload())

    assert len(calls) == 2
    assert result.decisions["u"].type == "noul"


def test_http_backend_does_not_retry_client_errors():
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(400, json={"error": "bad request"})

    from src.errors import RemoteBackendError

    backend = HttpSystemOneBackend(
        "http://remote", retries=1, transport=httpx.MockTransport(handler)
    )
    with pytest.raises(RemoteBackendError) as excinfo:
        backend.decide("state text", _retry_payload())

    assert len(calls) == 1
    assert excinfo.value.status_code == 400


def test_http_backend_retries_transport_error_once():
    calls = []

    def handler(request):
        calls.append(request)
        raise httpx.ConnectError("boom", request=request)

    from src.errors import RemoteBackendError

    backend = HttpSystemOneBackend(
        "http://remote", retries=1, transport=httpx.MockTransport(handler)
    )
    with pytest.raises(RemoteBackendError) as excinfo:
        backend.decide("secret state text", _retry_payload())

    assert len(calls) == 2
    assert "secret state text" not in str(excinfo.value)


def test_http_backend_zero_retries_fails_fast():
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(503, json={})

    from src.errors import RemoteBackendError

    backend = HttpSystemOneBackend(
        "http://remote", retries=0, transport=httpx.MockTransport(handler)
    )
    with pytest.raises(RemoteBackendError):
        backend.decide("state text", _retry_payload())

    assert len(calls) == 1


def test_http_backend_rejects_invalid_options():
    with pytest.raises(ValueError, match="timeout"):
        HttpSystemOneBackend("http://remote", timeout=0)
    with pytest.raises(ValueError, match="retries"):
        HttpSystemOneBackend("http://remote", retries=-1)


def test_create_backend_passes_http_budget(monkeypatch):
    from api.server import main

    monkeypatch.setenv("DECISION_BACKEND", "http")
    monkeypatch.setenv("DECISION_BACKEND_URL", "http://localhost:30000")
    monkeypatch.setenv("DECISION_BACKEND_TIMEOUT", "12.5")
    monkeypatch.setenv("DECISION_BACKEND_RETRIES", "2")
    backend = main.create_backend()

    assert backend.timeout == 12.5
    assert backend.retries == 2
