"""Тесты согласования размерности сервиса с реальным энкодером."""
import torch

from api.server.service import DecisionService
from src.sampler.questions import NoulSpec


class _DimEncoder:
    def __init__(self, dim: int):
        self.dim = dim

    def encode(self, states: list[str]) -> torch.Tensor:
        return torch.randn(len(states), self.dim)


def test_decide_uses_encoder_dim_for_heads():
    svc = DecisionService(encoder=_DimEncoder(dim=16), dim=64)
    result = svc.decide("state text", {"u": NoulSpec("urgent?")})
    assert "u" in result
    assert 0.0 <= result["u"].noul <= 1.0


def test_create_service_defaults_to_hash():
    from api.server import main
    from api.server.service import HashStateEncoder

    main.os.environ.pop("DECISION_MODEL_ID", None)
    svc = main.create_service()
    assert isinstance(svc.encoder, HashStateEncoder)


def test_create_service_wires_device_and_dtype(monkeypatch):
    from api.server import main
    from src.encoder import gemma as gemma_mod

    captured: dict = {}
    real_init = gemma_mod.GemmaStateEncoder.__init__

    def fake_init(self, *args, **kwargs):
        captured.update(kwargs)
        real_init(self, *args, **kwargs)

    monkeypatch.setattr(gemma_mod.GemmaStateEncoder, "__init__", fake_init)
    monkeypatch.setattr(gemma_mod.GemmaStateEncoder, "load", lambda self: self)
    monkeypatch.setenv("DECISION_MODEL_ID", "fake-model")
    monkeypatch.setenv("DECISION_DEVICE", "cpu")
    monkeypatch.setenv("DECISION_DTYPE", "bfloat16")

    main.create_service()

    assert captured["device"] == "cpu"
    assert captured["dtype"] == torch.bfloat16


def test_create_service_builds_encoder_kind(monkeypatch):
    from api.server import main
    from src.encoder import transformer as transformer_mod

    captured: dict = {}

    def fake_init(self, *args, **kwargs):
        captured.update(kwargs)
        self.model_id = kwargs.get("model_id", "bert-base-uncased")

    monkeypatch.setattr(
        transformer_mod.TransformerStateEncoder, "__init__", fake_init
    )
    monkeypatch.setattr(
        transformer_mod.TransformerStateEncoder, "load", lambda self: self
    )
    monkeypatch.setenv("DECISION_MODEL_ID", "bert-base-uncased")
    monkeypatch.setenv("DECISION_ENCODER_KIND", "encoder")
    monkeypatch.setenv("DECISION_DEVICE", "cpu")

    svc = main.create_service()

    assert isinstance(svc.encoder, transformer_mod.TransformerStateEncoder)
    assert captured["model_id"] == "bert-base-uncased"


def test_create_service_rejects_fp16_on_cpu(monkeypatch):
    import pytest

    from api.server import main

    monkeypatch.setenv("DECISION_MODEL_ID", "fake-model")
    monkeypatch.setenv("DECISION_DEVICE", "cpu")
    monkeypatch.setenv("DECISION_DTYPE", "float16")

    with pytest.raises(RuntimeError, match="float16"):
        main.create_service()


def test_create_vision_encoder_uses_resolved_device(monkeypatch):
    from api.server import main
    from src.encoder import vision as vision_mod

    captured: dict = {}

    def fake_init(self, *args, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(vision_mod.ClipVisionEncoder, "__init__", fake_init)
    monkeypatch.setattr(vision_mod.ClipVisionEncoder, "load", lambda self: self)
    monkeypatch.setenv("DECISION_VISION_MODEL_ID", "openai/clip-vit-base-patch32")
    monkeypatch.setenv("DECISION_DEVICE", "cpu")
    monkeypatch.delenv("DECISION_VISION_DIM", raising=False)
    monkeypatch.delenv("DECISION_MEDIA_ROOT", raising=False)

    main.create_vision_encoder()

    assert captured["device"] == "cpu"
    assert captured["dim"] == 512
    assert captured["model_id"] == "openai/clip-vit-base-patch32"
    assert captured["media_root"] is None


def test_create_vision_encoder_disabled_without_model_id(monkeypatch):
    from api.server import main

    monkeypatch.delenv("DECISION_VISION_MODEL_ID", raising=False)
    assert main.create_vision_encoder() is None


def test_parse_max_length_defaults_and_edges():
    import pytest

    from api.server.main import parse_max_length

    assert parse_max_length("256") == 256
    assert parse_max_length(" 128 ") == 128
    assert parse_max_length("none") is None
    assert parse_max_length("") is None
    assert parse_max_length("0") is None
    assert parse_max_length("-5") is None
    with pytest.raises(RuntimeError):
        parse_max_length("abc")


def test_service_warns_when_signature_checkpoint_missing(tmp_path, caplog):
    import logging

    from api.server.service import DecisionService
    from src.sampler.questions import NoulSpec

    svc = DecisionService(seed=0, checkpoint_dir=tmp_path)
    with caplog.at_level(logging.WARNING, logger="decision.service"):
        svc.decide("state", {"u": NoulSpec("urgent?")})
    assert any("serving untrained heads" in r.message for r in caplog.records)
    assert all(r.levelno >= logging.WARNING for r in caplog.records if "serving" in r.message)


def test_service_stays_quiet_when_checkpoint_dir_absent(tmp_path, caplog):
    import logging

    from api.server.service import DecisionService
    from src.sampler.questions import NoulSpec

    svc = DecisionService(seed=0, checkpoint_dir=tmp_path / "absent")
    with caplog.at_level(logging.INFO, logger="decision.service"):
        svc.decide("state", {"u": NoulSpec("urgent?")})
    assert not any(r.levelno >= logging.WARNING for r in caplog.records)
    assert any("no checkpoint dir" in r.message for r in caplog.records)


def test_spec_from_api_preserves_criteria():
    from eval.datasets.text_dataset import spec_from_api
    from src.sampler.questions import ChoiceSpec, NoulSpec, ScoreSpec

    noul = spec_from_api(
        {"type": "noul", "instructions": "u", "criteria": ["yes", "no"]}
    )
    assert isinstance(noul, NoulSpec) and noul.criteria == ("yes", "no")

    choice = spec_from_api(
        {
            "type": "choice",
            "instructions": "who",
            "options": ["a", "b"],
            "criteria": {"a": "x", "b": "y"},
        }
    )
    assert isinstance(choice, ChoiceSpec) and choice.criteria == {"a": "x", "b": "y"}

    score = spec_from_api(
        {
            "type": "score",
            "instructions": "how",
            "levels": ["lo", "hi"],
            "criteria": ["calm", "critical"],
        }
    )
    assert isinstance(score, ScoreSpec) and score.criteria == ("calm", "critical")

    plain = spec_from_api({"type": "noul", "instructions": "u"})
    assert plain.criteria is None


def test_train_heads_encoder_respects_device_env(monkeypatch):
    from src.encoder import gemma as gemma_mod
    from training.recipe import train_heads

    captured: dict = {}
    monkeypatch.setattr(
        gemma_mod.GemmaStateEncoder,
        "__init__",
        lambda self, *a, **k: captured.update(k),
    )
    monkeypatch.setattr(gemma_mod.GemmaStateEncoder, "load", lambda self: self)
    monkeypatch.setenv("DECISION_DEVICE", "cpu")
    monkeypatch.setenv("DECISION_DTYPE", "bfloat16")

    train_heads._make_encoder("openai-community/gpt2", 64)

    assert captured["device"] == "cpu"
    assert captured["dtype"] == torch.bfloat16