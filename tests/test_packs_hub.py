"""Паки с Hugging Face Hub: спецификация, кэш, проверка манифеста, wiring."""
from __future__ import annotations

import hashlib
import json

import pytest

from src import packs_hub


def _write_pack(root):
    pack = root / "packs" / "public-gpt2"
    pack.mkdir(parents=True)
    files = {
        "heads.pt": b"heads",
        "metadata.json": b"{}",
        "temperature.json": b"{}",
    }
    for name, payload in files.items():
        (pack / name).write_bytes(payload)
    manifest = {
        "schema_version": "1",
        "library": "decision-cascade",
        "library_version": "1.7.0",
        "packs": [
            {
                "name": "public-gpt2",
                "files": {
                    name: {
                        "sha256": hashlib.sha256(payload).hexdigest(),
                        "bytes": len(payload),
                    }
                    for name, payload in files.items()
                },
            }
        ],
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_parse_source_defaults_to_main_and_validates():
    assert packs_hub.parse_source("owner/name") == packs_hub.PackSource(
        "owner/name", "main"
    )
    assert packs_hub.parse_source("owner/name@v1.5.0") == packs_hub.PackSource(
        "owner/name", "v1.5.0"
    )
    for bad in ("owner", "owner/name/extra", "/name", "owner/", ""):
        with pytest.raises(ValueError, match="owner/name"):
            packs_hub.parse_source(bad)


def test_resolve_downloads_once_then_uses_cache(tmp_path):
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        _write_pack(tmp_path / kwargs["local_dir"])

    first = packs_hub.resolve("owner/name@v1", root=tmp_path, download=download)
    second = packs_hub.resolve("owner/name@v1", root=tmp_path, download=download)

    assert first == second == tmp_path / "owner--name" / "v1" / "packs"
    assert (first / "public-gpt2" / "heads.pt").is_file()
    assert len(calls) == 1
    assert calls[0]["repo_id"] == "owner/name"
    assert calls[0]["revision"] == "v1"


def test_resolve_redownloads_broken_cache(tmp_path):
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        _write_pack(tmp_path / kwargs["local_dir"])

    path = packs_hub.resolve("owner/name@v1", root=tmp_path, download=download)
    (path / "public-gpt2" / "heads.pt").write_bytes(b"tampered")

    again = packs_hub.resolve("owner/name@v1", root=tmp_path, download=download)

    assert again == path
    assert (again / "public-gpt2" / "heads.pt").read_bytes() == b"heads"
    assert len(calls) == 2


def test_resolve_rejects_bad_download(tmp_path):
    def download(**kwargs):
        target = tmp_path / kwargs["local_dir"]
        _write_pack(target)
        (target / "packs" / "public-gpt2" / "heads.pt").write_bytes(b"tampered")

    with pytest.raises(RuntimeError, match="sha256 mismatch"):
        packs_hub.resolve("owner/name@v1", root=tmp_path, download=download)


def test_verify_rejects_unsafe_pack_name(tmp_path):
    _write_pack(tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    manifest["packs"][0]["name"] = "../escape"
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RuntimeError, match="unsafe pack name"):
        packs_hub.verify_manifest(tmp_path)


def test_create_service_uses_pack_repo(monkeypatch, tmp_path):
    from api.server.main import create_service

    resolved = tmp_path / "packs"
    resolved.mkdir()
    monkeypatch.setenv("DECISION_PACK_REPO", "owner/name@v1")
    monkeypatch.delenv("DECISION_CHECKPOINT_DIR", raising=False)
    monkeypatch.setattr(packs_hub, "resolve", lambda value: resolved)

    service = create_service()

    assert service.checkpoint_dir == resolved


def test_create_service_rejects_both_sources(monkeypatch, tmp_path):
    from api.server.main import create_service

    monkeypatch.setenv("DECISION_PACK_REPO", "owner/name")
    monkeypatch.setenv("DECISION_CHECKPOINT_DIR", str(tmp_path))

    with pytest.raises(RuntimeError, match="either DECISION_PACK_REPO"):
        create_service()


def test_create_service_wraps_pack_repo_errors(monkeypatch):
    from api.server.main import create_service

    monkeypatch.setenv("DECISION_PACK_REPO", "owner/name")
    monkeypatch.delenv("DECISION_CHECKPOINT_DIR", raising=False)
    monkeypatch.setattr(
        packs_hub, "resolve", lambda value: (_ for _ in ()).throw(RuntimeError("boom"))
    )

    with pytest.raises(RuntimeError, match="DECISION_PACK_REPO"):
        create_service()
