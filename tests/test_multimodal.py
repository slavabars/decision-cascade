"""Тесты локальной мультимодальности: текст + изображения/видео — RED до реализации."""
import pytest

from api.server.service import DecisionService
from src.encoder.vision import HashVisionEncoder
from src.sampler.questions import NoulSpec


def _service_with_vision():
    return DecisionService(seed=0, vision_encoder=HashVisionEncoder(dim=64))


def test_modalities_text_only_without_vision():
    assert DecisionService(seed=0).modalities == frozenset({"text"})


def test_modalities_include_media_with_vision():
    assert _service_with_vision().modalities == frozenset({"text", "image", "video"})


def test_decide_with_images_changes_result():
    svc = _service_with_vision()
    questions = {"u": NoulSpec("urgent?")}
    text_only = svc.decide("state", questions)["u"].noul
    with_image = svc.decide("state", questions, images=["a screenshot"])["u"].noul
    assert with_image != pytest.approx(text_only)


def test_decide_with_videos_changes_result():
    svc = _service_with_vision()
    questions = {"u": NoulSpec("urgent?")}
    text_only = svc.decide("state", questions)["u"].noul
    with_video = svc.decide("state", questions, videos=["clip"])["u"].noul
    assert with_video != pytest.approx(text_only)


def test_media_weight_zero_matches_text_only():
    svc = _service_with_vision()
    questions = {"u": NoulSpec("urgent?")}
    text_only = svc.decide("state", questions)["u"].noul
    neutralized = svc.decide(
        "state",
        questions,
        images=["a screenshot"],
        media_kwargs={"media_weight": 0.0},
    )["u"].noul
    assert neutralized == pytest.approx(text_only)


def test_decide_with_media_without_vision_raises():
    svc = DecisionService(seed=0)
    with pytest.raises(ValueError, match="vision"):
        svc.decide("state", {"u": NoulSpec("?")}, images=["x"])


def test_decide_rejects_video_when_encoder_is_image_only():
    class ImageOnly(HashVisionEncoder):
        modalities = frozenset({"image"})

    svc = DecisionService(seed=0, vision_encoder=ImageOnly(dim=16))
    assert svc.modalities == frozenset({"text", "image"})
    with pytest.raises(ValueError, match="video"):
        svc.decide("state", {"u": NoulSpec("?")}, videos=["v"])
    assert svc.decide("state", {"u": NoulSpec("?")}, images=["i"])["u"].type == "noul"


def test_media_kwargs_max_frames_forwarded():
    class SpyVision(HashVisionEncoder):
        def __init__(self):
            super().__init__(dim=8)
            self.calls: list[int] = []

        def encode_videos(self, videos, *, max_frames=8):
            self.calls.append(max_frames)
            return super().encode_videos(videos, max_frames=max_frames)

    spy = SpyVision()
    svc = DecisionService(seed=0, vision_encoder=spy)
    svc.decide(
        "state",
        {"u": NoulSpec("?")},
        videos=["clip"],
        media_kwargs={"max_frames": 2},
    )
    assert spy.calls == [2]
