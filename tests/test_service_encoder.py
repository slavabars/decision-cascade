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