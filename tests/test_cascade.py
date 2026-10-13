"""Тесты каскадного бэкенда (локально-если-уверены → внешняя) — RED до реализации."""
import pytest

from src.backends.base import IMAGE, TEXT, BackendResult, Capabilities
from src.backends.cascade import CascadeBackend
from src.schemas import NoulResult


class _StubBackend:
    def __init__(self, name, confidence, modalities=(TEXT,)):
        self.name = name
        self._confidence = confidence
        self.capabilities = Capabilities(modalities=frozenset(modalities))
        self.calls: list[tuple] = []
        self.closed = False

    def decide(self, state, questions, *, images=None, videos=None, media_kwargs=None):
        self.calls.append((state, tuple(questions)))
        decisions = {
            name: NoulResult(noul=self._confidence, confidence=self._confidence)
            for name in questions
        }
        return BackendResult(decisions=decisions, usage={"calls": 1})

    def close(self):
        self.closed = True


def _questions(names):
    from src.sampler.questions import NoulSpec

    return {name: NoulSpec(f"?{name}") for name in names}


def test_confident_primary_skips_fallback():
    primary = _StubBackend("local", confidence=0.9)
    fallback = _StubBackend("remote", confidence=0.9)
    cascade = CascadeBackend(primary, fallback, threshold=0.6)

    result = cascade.decide("s", _questions(["a", "b"]))

    assert set(result.decisions) == {"a", "b"}
    assert fallback.calls == []
    assert result.usage["escalated"] == 0


def test_uncertain_primary_escalates_subset():
    primary = _StubBackend("local", confidence=0.3)
    fallback = _StubBackend("remote", confidence=0.95)
    cascade = CascadeBackend(primary, fallback, threshold=0.6)

    result = cascade.decide("s", _questions(["a", "b"]))

    assert fallback.calls == [("s", ("a", "b"))]
    assert all(r.confidence == 0.95 for r in result.decisions.values())
    assert result.usage["escalated"] == 2


def test_partial_escalation_only_uncertain():
    primary = _StubBackend("local", confidence=0.9)
    fallback = _StubBackend("remote", confidence=0.95)
    cascade = CascadeBackend(primary, fallback, threshold=0.6)
    cascade._confidence_of = None  # ensure default path used

    # make one question uncertain by overriding primary result per question
    from src.schemas import NoulResult as _R

    def decide(state, questions, *, images=None, videos=None, media_kwargs=None):
        primary.calls.append((state, tuple(questions)))
        return BackendResult(
            decisions={
                "a": _R(noul=0.95, confidence=0.9),
                "b": _R(noul=0.55, confidence=0.1),
            },
            usage={},
        )

    primary.decide = decide
    result = cascade.decide("s", _questions(["a", "b"]))
    assert fallback.calls == [("s", ("b",))]
    assert result.decisions["a"].confidence == 0.9
    assert result.decisions["b"].confidence == 0.95
    assert result.usage["escalated"] == 1


def test_missing_modality_goes_straight_to_fallback():
    primary = _StubBackend("local", confidence=0.9, modalities=(TEXT,))
    fallback = _StubBackend("remote", confidence=0.9, modalities=(TEXT, IMAGE))
    cascade = CascadeBackend(primary, fallback, threshold=0.6)

    result = cascade.decide("s", _questions(["a"]), images=["x.png"])

    assert primary.calls == []
    assert fallback.calls == [("s", ("a",))]
    assert result.usage["escalated"] == 1


def test_capabilities_union():
    primary = _StubBackend("local", 0.9, modalities=(TEXT,))
    fallback = _StubBackend("remote", 0.9, modalities=(TEXT, IMAGE))
    cascade = CascadeBackend(primary, fallback, threshold=0.6)
    assert cascade.capabilities.modalities == frozenset({TEXT, IMAGE})
    assert cascade.name == "cascade[local->remote]"


def test_close_closes_both():
    primary = _StubBackend("local", 0.9)
    fallback = _StubBackend("remote", 0.9)
    CascadeBackend(primary, fallback).close()
    assert primary.closed and fallback.closed


def test_create_backend_cascade_requires_url(monkeypatch):
    from api.server import main

    monkeypatch.setenv("DECISION_BACKEND", "cascade")
    monkeypatch.delenv("DECISION_CASCADE_FALLBACK_URL", raising=False)
    with pytest.raises(RuntimeError):
        main.create_backend(service=main.DecisionService())


def test_create_backend_cascade_wires(monkeypatch):
    from api.server import main

    monkeypatch.setenv("DECISION_BACKEND", "cascade")
    monkeypatch.setenv("DECISION_CASCADE_FALLBACK_URL", "http://127.0.0.1:9")
    backend = main.create_backend(service=main.DecisionService())
    assert isinstance(backend, CascadeBackend)
    backend.close()
