"""Реальные тесты CLIP vision-энкодера (opt-in, качают модель с HF).

Обычный ``pytest`` их пропускает; чтобы прогнать на лёгкой CLIP:

    DECISION_RUN_MODEL_TESTS=1 .venv/bin/python -m pytest -m model -v

Модель по умолчанию — ``openai/clip-vit-base-patch32`` (~0.6 ГБ), переопределяется
через ``DECISION_TEST_VISION_MODEL``.
"""
import os

import pytest
import torch
from PIL import Image

from src.encoder.vision import ClipVisionEncoder, VisionEncoder

MODEL_ID = os.environ.get(
    "DECISION_TEST_VISION_MODEL", "openai/clip-vit-base-patch32"
)
RUN_MODEL_TESTS = os.environ.get("DECISION_RUN_MODEL_TESTS") == "1"

pytestmark = [
    pytest.mark.model,
    pytest.mark.skipif(
        not RUN_MODEL_TESTS,
        reason="set DECISION_RUN_MODEL_TESTS=1 to run real-model tests",
    ),
]


def _save(path, mode: str, size=(224, 224)) -> str:
    if mode == "black":
        image = Image.new("RGB", size, (0, 0, 0))
    elif mode == "white":
        image = Image.new("RGB", size, (255, 255, 255))
    elif mode == "noise":
        generator = torch.Generator().manual_seed(1234)
        pixels = (torch.rand(3, size[1], size[0], generator=generator) * 255).to(
            torch.uint8
        )
        image = Image.fromarray(pixels.permute(1, 2, 0).numpy(), mode="RGB")
    else:  # pragma: no cover - защита от опечатки
        raise ValueError(mode)
    image.save(path)
    return str(path)


@pytest.fixture(scope="module")
def encoder() -> ClipVisionEncoder:
    enc = ClipVisionEncoder(model_id=MODEL_ID, dim=512)
    enc.load()
    return enc


def test_clip_encoder_satisfies_protocol(encoder):
    assert isinstance(encoder, VisionEncoder)
    assert encoder.modalities == frozenset({"image"})


def test_encode_images_shape_and_determinism(encoder, tmp_path):
    black = _save(tmp_path / "black.png", "black")
    noise = _save(tmp_path / "noise.png", "noise")

    first = encoder.encode_images([black, noise])
    second = encoder.encode_images([black, noise])

    assert first.shape == (2, 512)
    assert first.device.type == "cpu"
    assert torch.allclose(first, second, atol=1e-6)
    assert not torch.allclose(first[0], first[1], atol=1e-3)


def test_identical_images_are_stable_and_batch_invariant(encoder, tmp_path):
    image = _save(tmp_path / "white.png", "white")
    single = encoder.encode_images([image])
    batched = encoder.encode_images([image, image])

    assert single.shape == (1, 512)
    assert torch.allclose(batched[0], batched[1], atol=1e-5)
    # Одна и та же картинка в батче 1 и батче 2 даёт почти один вектор.
    cosine = torch.nn.functional.cosine_similarity(single[0], batched[0], dim=0)
    assert cosine.item() > 0.999


def test_encode_empty_returns_zero_rows(encoder):
    out = encoder.encode_images([])
    assert out.shape == (0, 512)


def test_projection_to_smaller_dim(encoder, tmp_path):
    black = _save(tmp_path / "black.png", "black")
    noise = _save(tmp_path / "noise.png", "noise")

    projected = ClipVisionEncoder(model_id=MODEL_ID, dim=64)
    # Переиспользуем уже загруженную модель, чтобы не качать/грузить второй раз.
    projected._model = encoder._model
    projected._processor = encoder._processor
    projected.device = encoder.device

    out = projected.encode_images([black, noise])
    assert out.shape == (2, 64)
    assert not torch.allclose(out[0], out[1], atol=1e-3)


def test_real_images_through_decision_service(encoder, tmp_path):
    from api.server.service import DecisionService
    from src.sampler.questions import NoulSpec

    black = _save(tmp_path / "black.png", "black")
    service = DecisionService(seed=0, vision_encoder=encoder)
    assert "image" in service.modalities

    result = service.decide(
        "обращение клиента", {"urgent": NoulSpec("срочно?")}, images=[black]
    )
    assert "urgent" in result


def test_encode_videos_is_not_supported(encoder):
    with pytest.raises(NotImplementedError):
        encoder.encode_videos(["anything.mp4"])
