"""Тесты vision-энкодера (offline fallback) — написаны до реализации (TDD RED)."""
import pytest
import torch

from src.encoder.vision import HashVisionEncoder


def test_encode_images_shape_and_determinism():
    enc = HashVisionEncoder(dim=32)
    a = enc.encode_images(["foo", "bar"])
    b = enc.encode_images(["foo", "bar"])
    assert a.shape == (2, 32)
    assert torch.allclose(a, b)
    assert not torch.allclose(a[0], a[1])


def test_encode_images_are_normalized():
    enc = HashVisionEncoder(dim=16)
    out = enc.encode_images(["hello world"])
    assert torch.allclose(out.norm(dim=-1), torch.ones(1), atol=1e-5)


def test_encode_images_empty_returns_zero_rows():
    enc = HashVisionEncoder(dim=8)
    out = enc.encode_images([])
    assert out.shape == (0, 8)


def test_encode_images_reads_file_contents(tmp_path):
    raw = b"\x00\x01\x02\xff\xfe" * 16
    path = tmp_path / "img.bin"
    path.write_bytes(raw)
    enc = HashVisionEncoder(dim=16)
    from_file = enc.encode_images([str(path)])
    from_bytes = enc.encode_images([raw])
    assert from_file.shape == (1, 16)
    assert torch.allclose(from_file, from_bytes)
    assert not torch.allclose(from_file, enc.encode_images([b"other"]))


def test_encode_videos_shape_and_determinism():
    enc = HashVisionEncoder(dim=24)
    a = enc.encode_videos(["clip-a", "clip-b"], max_frames=3)
    b = enc.encode_videos(["clip-a", "clip-b"], max_frames=3)
    assert a.shape == (2, 24)
    assert torch.allclose(a, b)


def test_encode_videos_max_frames_changes_result():
    enc = HashVisionEncoder(dim=24)
    data = b"frame-data" * 100
    one = enc.encode_videos([data], max_frames=1)
    many = enc.encode_videos([data], max_frames=4)
    assert one.shape == many.shape == (1, 24)
    assert not torch.allclose(one, many)


def test_hash_vision_encoder_satisfies_protocol():
    from src.encoder.vision import VisionEncoder

    assert isinstance(HashVisionEncoder(dim=8), VisionEncoder)


# --- ClipVisionEncoder: извлечение признаков без загрузки модели (unit) ---


class _FakeBatch(dict):
    def to(self, device):
        return self


class _FakeProcessor:
    def __call__(self, images=None, return_tensors=None):
        return _FakeBatch(pixel_values=torch.zeros(len(images), 3, 4, 4))


class _OutputObject:
    """Имитация BaseModelOutputWithPooling из transformers >=5."""

    def __init__(self, features):
        self.pooler_output = features
        self.last_hidden_state = features.unsqueeze(1)


class _ObjectClipModel:
    def get_image_features(self, **inputs):
        n = inputs["pixel_values"].shape[0]
        return _OutputObject(torch.ones(n, 512))


class _TensorClipModel:
    def get_image_features(self, **inputs):
        n = inputs["pixel_values"].shape[0]
        return torch.ones(n, 512)


def _fake_encoder(dim: int, model) -> "object":
    from src.encoder.vision import ClipVisionEncoder

    encoder = ClipVisionEncoder(model_id="fake", dim=dim, device="cpu")
    encoder._processor = _FakeProcessor()
    encoder._model = model
    return encoder


def test_clip_encoder_reads_features_from_output_object(tmp_path):
    from PIL import Image

    image = tmp_path / "x.png"
    Image.new("RGB", (8, 8), (1, 2, 3)).save(image)

    encoder = _fake_encoder(512, _ObjectClipModel())
    out = encoder.encode_images([str(image), str(image)])

    assert out.shape == (2, 512)
    assert out.dtype == torch.float32


def test_clip_encoder_reads_features_from_tensor(tmp_path):
    from PIL import Image

    image = tmp_path / "x.png"
    Image.new("RGB", (8, 8), (1, 2, 3)).save(image)

    encoder = _fake_encoder(512, _TensorClipModel())
    out = encoder.encode_images([str(image)])

    assert out.shape == (1, 512)


def test_clip_encoder_projects_to_requested_dim(tmp_path):
    from PIL import Image

    image = tmp_path / "x.png"
    Image.new("RGB", (8, 8), (1, 2, 3)).save(image)

    encoder = _fake_encoder(64, _TensorClipModel())
    out = encoder.encode_images([str(image)])

    assert out.shape == (1, 64)


def test_clip_encoder_requires_load_before_encode(tmp_path):
    from PIL import Image

    from src.encoder.vision import ClipVisionEncoder

    image = tmp_path / "x.png"
    Image.new("RGB", (8, 8), (1, 2, 3)).save(image)
    encoder = ClipVisionEncoder(model_id="fake", dim=512, device="cpu")
    with pytest.raises(RuntimeError, match="load"):
        encoder.encode_images([str(image)])


def test_clip_encoder_load_uses_auto_backbone(monkeypatch):
    # load() должен быть архитектурно-нейтральным: бэкбон грузится через
    # AutoModel/AutoProcessor, поэтому подходит и SigLIP, и CLIP.
    from src.encoder import vision

    captured = {}

    class _FakeModel:
        def to(self, device):
            captured["device"] = device
            return self

        def eval(self):
            captured["eval"] = True
            return self

    def _fake_backbone(model_id):
        captured["model_id"] = model_id
        return "processor", _FakeModel()

    monkeypatch.setattr(vision, "_load_vision_backbone", _fake_backbone)

    model_id = "google/siglip-base-patch16-224"
    encoder = vision.ClipVisionEncoder(model_id=model_id, dim=768, device="cpu")
    encoder.load()

    assert captured == {"model_id": model_id, "device": "cpu", "eval": True}
    assert encoder._processor == "processor"
    assert isinstance(encoder._model, _FakeModel)
