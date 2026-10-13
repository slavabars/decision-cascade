"""Тесты vision-энкодера (offline fallback) — написаны до реализации (TDD RED)."""
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
