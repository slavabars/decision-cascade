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