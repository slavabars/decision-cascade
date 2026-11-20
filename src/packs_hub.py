"""Паки весов с Hugging Face Hub: `DECISION_PACK_REPO` и проверка кэша.

Пак на Хабе — каталог ``packs/<имя>/`` с ``heads.pt``, ``metadata.json`` и
``temperature.json`` плюс ``manifest.json`` с sha256 каждого файла. Здесь
разрешается спецификация ``owner/name[@revision]`` в локальный каталог паков:

- если в кэше уже есть манифест и хэши сходятся, сеть не нужна;
- иначе каталог скачивается через ``snapshot_download`` и проверяется;
- битый кэш перекачивается один раз, а не подсовывается сервису.

Модуль намеренно не импортирует ``huggingface_hub`` на уровне модуля: юнит-тесты
подставляют загрузчик, а сервис без ``DECISION_PACK_REPO`` вообще не трогает Hub.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "1"
DEFAULT_CACHE = Path.home() / ".cache" / "decision-cascade" / "packs"
ALLOW_PATTERNS = ["manifest.json", "packs/*/*"]


@dataclass(frozen=True)
class PackSource:
    """Репозиторий и ревизия: ``owner/name`` и тег (по умолчанию ``main``)."""

    repo: str
    revision: str


def parse_source(value: str) -> PackSource:
    """Разобрать ``owner/name[@revision]``; мусор — ошибка, а не догадка."""
    repo, _, revision = value.strip().partition("@")
    if repo.count("/") != 1 or not all(repo.split("/")):
        raise ValueError(
            f"DECISION_PACK_REPO must look like owner/name[@revision], got {value!r}"
        )
    return PackSource(repo=repo, revision=revision or "main")


def cache_dir() -> Path:
    """Каталог кэша: ``DECISION_PACK_CACHE`` или ``~/.cache/decision-cascade/packs``."""
    value = os.environ.get("DECISION_PACK_CACHE")
    return Path(value).expanduser() if value else DEFAULT_CACHE


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_manifest(root: Path) -> dict[str, Any]:
    """Проверить манифест и все файлы паков; вернуть разобранный манифест."""
    manifest_path = root / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeError(f"manifest.json is missing in {root}") from exc
    except ValueError as exc:
        raise RuntimeError(f"manifest.json is not valid JSON: {exc}") from exc
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError(
            "unsupported manifest schema_version: "
            f"{manifest.get('schema_version')!r}"
        )
    packs = manifest.get("packs")
    if not isinstance(packs, list) or not packs:
        raise RuntimeError("manifest has no packs")
    for pack in packs:
        name = pack.get("name")
        if (
            not isinstance(name, str)
            or not name
            or "/" in name
            or "\\" in name
            or name in {".", ".."}
        ):
            raise RuntimeError(f"unsafe pack name in manifest: {name!r}")
        files = pack.get("files")
        if not isinstance(files, dict) or not files:
            raise RuntimeError(f"pack {name!r} has no files in manifest")
        for file_name, info in files.items():
            path = root / "packs" / name / file_name
            if not path.is_file():
                raise RuntimeError(f"pack file is missing: {path}")
            if sha256(path) != info.get("sha256"):
                raise RuntimeError(f"sha256 mismatch: {path}")
    return manifest


def _snapshot_download(**kwargs: Any) -> None:
    """Загрузка с Hub; импорт ленивый, чтобы не тянуть сеть в тестах."""
    from huggingface_hub import snapshot_download

    snapshot_download(**kwargs)


def resolve(
    value: str,
    *,
    root: Path | None = None,
    download: Callable[..., None] | None = None,
) -> Path:
    """Спецификация пака -> локальный каталог ``packs`` (сеть только при промахе)."""
    source = parse_source(value)
    base = root or cache_dir()
    target = base / source.repo.replace("/", "--") / source.revision
    downloader = download or _snapshot_download

    if (target / "manifest.json").is_file():
        try:
            verify_manifest(target)
            logger.info("packs: using cached %s@%s", source.repo, source.revision)
            return target / "packs"
        except RuntimeError as exc:
            logger.warning(
                "packs: cache %s is invalid (%s); re-downloading", target, exc
            )
            shutil.rmtree(target)

    target.mkdir(parents=True, exist_ok=True)
    downloader(
        repo_id=source.repo,
        revision=source.revision,
        local_dir=str(target),
        allow_patterns=list(ALLOW_PATTERNS),
    )
    verify_manifest(target)
    return target / "packs"
