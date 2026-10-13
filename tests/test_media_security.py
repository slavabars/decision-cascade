"""Тесты ограничения доступа к медиафайлам — написаны до реализации (RED)."""
import pytest

from src.encoder.vision import (
    ClipVisionEncoder,
    HashVisionEncoder,
    _read_source,
    _resolve_media_path,
)


def test_read_source_within_root(tmp_path):
    root = tmp_path / "media"
    root.mkdir()
    (root / "a.png").write_bytes(b"hello")
    assert _read_source("a.png", media_root=root) == b"hello"


def test_read_source_rejects_escape(tmp_path):
    root = tmp_path / "media"
    root.mkdir()
    (tmp_path / "secret.txt").write_bytes(b"secret")
    with pytest.raises(ValueError):
        _read_source("../secret.txt", media_root=root)


def test_read_source_absolute_outside_root_rejected(tmp_path):
    root = tmp_path / "media"
    root.mkdir()
    outside = tmp_path / "secret.txt"
    outside.write_bytes(b"secret")
    with pytest.raises(ValueError):
        _read_source(str(outside), media_root=root)


def test_read_source_bytes_always_allowed(tmp_path):
    root = tmp_path / "media"
    root.mkdir()
    assert _read_source(b"raw", media_root=root) == b"raw"


def test_hash_vision_encoder_honors_media_root(tmp_path):
    root = tmp_path / "media"
    root.mkdir()
    (root / "a.png").write_bytes(b"abc")
    (tmp_path / "secret.txt").write_bytes(b"secret")
    enc = HashVisionEncoder(dim=8, media_root=root)
    enc.encode_images(["a.png"])
    with pytest.raises(ValueError):
        enc.encode_images(["../secret.txt"])


def test_clip_encoder_resolves_within_root(tmp_path):
    root = tmp_path / "media"
    root.mkdir()
    (root / "a.png").write_bytes(b"abc")
    enc = ClipVisionEncoder(dim=8, media_root=root)
    assert enc._resolve("a.png") == (root / "a.png").resolve()
    with pytest.raises(ValueError):
        enc._resolve("../secret.txt")


def test_resolve_media_path_returns_resolved(tmp_path):
    root = tmp_path / "media"
    root.mkdir()
    target = root / "sub" / "a.png"
    target.parent.mkdir()
    target.write_bytes(b"x")
    assert _resolve_media_path("sub/a.png", root) == target.resolve()


def test_parse_dim_bounds():
    import pytest as _pytest

    from api.server.main import parse_dim

    assert parse_dim("DECISION_FEATURE_DIM", "768", default=768, lo=1, hi=8192) == 768
    assert parse_dim("X", None, default=64, lo=1, hi=8192) == 64
    with _pytest.raises(RuntimeError):
        parse_dim("X", "0", default=64, lo=1, hi=8192)
    with _pytest.raises(RuntimeError):
        parse_dim("X", "100000", default=64, lo=1, hi=8192)
    with _pytest.raises(RuntimeError):
        parse_dim("X", "abc", default=64, lo=1, hi=8192)
