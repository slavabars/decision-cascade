"""Vision-энкодеры: изображения/видео -> батч признаков.

По аналогии с текстовым ``HashStateEncoder`` есть детерминированный offline
fallback ``HashVisionEncoder`` (без загрузки моделей) и реальный
``ClipVisionEncoder`` на transformers (загружается отдельно, ``load()``).
Любой объект, реализующий ``VisionEncoder``, можно передать в
``DecisionService(vision_encoder=...)`` — сервис сам объявит модальности.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Protocol, runtime_checkable

import torch
from torch import Tensor

VISION_FEATURE_DIM = 64
_MAX_BYTES = 1_000_000

MediaRef = str | bytes


@runtime_checkable
class VisionEncoder(Protocol):
    dim: int
    modalities: frozenset[str]

    def encode_images(self, images: list[MediaRef]) -> Tensor:
        ...

    def encode_videos(
        self, videos: list[MediaRef], *, max_frames: int = 8
    ) -> Tensor:
        ...


def _resolve_media_path(ref: MediaRef, media_root: Path) -> Path:
    """Разрешает путь к медиа внутри allow-listed корня и запрещает выход наружу."""
    if isinstance(ref, bytes):
        raise TypeError("media reference is bytes, not a path")
    root = Path(media_root).resolve()
    raw = Path(ref)
    candidate = raw.resolve() if raw.is_absolute() else (root / raw).resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError(f"media path {ref!r} escapes media root {root}")
    return candidate


def _read_source(ref: MediaRef, media_root: Path | None = None) -> bytes:
    """Читает байты из файла, если путь существует, иначе трактует ref как данные.

    При заданном ``media_root`` чтение из файловой системы разрешено только
    внутри этого каталога — защита от чтения произвольных путей.
    """
    if isinstance(ref, bytes):
        return ref
    if media_root is not None:
        path = _resolve_media_path(ref, media_root)
        if path.is_file():
            return path.read_bytes()[:_MAX_BYTES]
        return ref.encode("utf-8")
    try:
        path = Path(ref)
        if path.is_file():
            with path.open("rb") as handle:
                return handle.read(_MAX_BYTES)
    except (OSError, ValueError):
        pass
    return ref.encode("utf-8")


def _extract_image_features(outputs: object) -> Tensor:
    """Достаёт тензор признаков изображения из результата ``get_image_features``.

    В transformers >=5 ``get_image_features`` (CLIP/SigLIP) возвращает
    ``BaseModelOutputWithPooling`` (уже спроецированный вектор лежит в
    ``pooler_output``), в старых версиях — сразу тензор. Поддерживаем оба, плюс
    ``image_embeds`` и усреднение ``last_hidden_state`` как запасной вариант.
    """
    if isinstance(outputs, Tensor):
        return outputs
    for attr in ("pooler_output", "image_embeds"):
        value = getattr(outputs, attr, None)
        if isinstance(value, Tensor):
            return value
    hidden = getattr(outputs, "last_hidden_state", None)
    if isinstance(hidden, Tensor):
        return hidden.mean(dim=1)
    raise RuntimeError(
        f"cannot extract image features from {type(outputs).__name__}"
    )


class HashVisionEncoder(VisionEncoder):
    """Детерминированный fallback: хэш содержимого в вектор признаков.

    Не требует моделей и работает офлайн — годится для тестов и для сервисов,
    где достаточно «поймать» различие входов. Для реального качества
    подключайте ``ClipVisionEncoder``.
    """

    modalities = frozenset({"image", "video"})

    def __init__(
        self,
        dim: int = VISION_FEATURE_DIM,
        max_bytes: int = _MAX_BYTES,
        media_root: Path | None = None,
    ):
        self.dim = dim
        self.max_bytes = max_bytes
        self.media_root = Path(media_root) if media_root is not None else None

    def _vector(self, data: bytes) -> Tensor:
        vector = torch.zeros(self.dim)
        for start in range(0, len(data), 64):
            chunk = data[start : start + 64]
            digest = hashlib.blake2b(chunk, digest_size=8).digest()
            vector[int.from_bytes(digest, "big") % self.dim] += 1.0
        norm = vector.norm()
        return vector / norm if norm > 0 else vector

    def encode_images(self, images: list[MediaRef]) -> Tensor:
        rows = [
            self._vector(_read_source(image, self.media_root)[: self.max_bytes])
            for image in images
        ]
        if not rows:
            return torch.zeros(0, self.dim)
        return torch.stack(rows)

    def encode_videos(
        self, videos: list[MediaRef], *, max_frames: int = 8
    ) -> Tensor:
        if max_frames < 1:
            raise ValueError("max_frames must be >= 1")
        rows = []
        for video in videos:
            data = _read_source(video, self.media_root)[: self.max_bytes]
            segment = max(1, len(data) // max_frames)
            frame_vectors = [
                self._vector(data[start : start + segment])
                for start in range(0, len(data), segment)
            ][:max_frames]
            stacked = torch.stack(frame_vectors) if frame_vectors else torch.zeros(
                1, self.dim
            )
            rows.append(stacked.mean(dim=0))
        if not rows:
            return torch.zeros(0, self.dim)
        return torch.stack(rows)


def _load_vision_backbone(model_id: str) -> tuple[object, object]:
    """Загружает vision-бэкбон через ``AutoModel``/``AutoImageProcessor``.

    Для признаков изображений нужен только image-процессор, поэтому токенизатор
    не грузится (и его опциональные зависимости — например, ``sentencepiece``
    у SigLIP — не требуются). Подходят и CLIP (``openai/clip-vit-*``), и SigLIP
    (``google/siglip-*``): у обеих архитектур есть ``get_image_features``.
    Возвращает ``(processor, model)``.
    """
    from transformers import AutoImageProcessor, AutoModel

    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id)
    return processor, model


class ClipVisionEncoder(VisionEncoder):  # pragma: no cover - требует загрузки модели
    """Реальный vision-энкодер на CLIP/SigLIP (transformers, опционально).

    Загрузка идёт через ``AutoModel``/``AutoImageProcessor``, поэтому подходят и
    CLIP (``openai/clip-vit-*``), и SigLIP (``google/siglip-*``) — у обеих
    архитектур есть ``get_image_features``. Поддерживаются только изображения:
    видео требует отдельного декодирования кадров (см. ``encode_images``).
    """

    modalities = frozenset({"image"})

    def __init__(
        self,
        model_id: str = "openai/clip-vit-base-patch32",
        dim: int = 512,
        device: str | None = None,
        media_root: Path | None = None,
    ):
        self.model_id = model_id
        self.dim = dim
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.media_root = Path(media_root) if media_root is not None else None
        self._model = None
        self._processor = None

    def _resolve(self, ref: MediaRef) -> MediaRef:
        """Разрешает путь внутри allow-listed корня (если он задан)."""
        if isinstance(ref, bytes) or self.media_root is None:
            return Path(ref) if isinstance(ref, str) else ref
        return _resolve_media_path(ref, self.media_root)

    def load(self) -> ClipVisionEncoder:
        # Бэкбон грузится через Auto* (CLIP и SigLIP), затем переносится на
        # устройство и переводится в eval-режим.
        self._processor, self._model = _load_vision_backbone(self.model_id)
        self._model = self._model.to(self.device).eval()
        return self

    def _encode_pil(self, images: list[object]) -> Tensor:
        if self._model is None:
            raise RuntimeError("ClipVisionEncoder.load() must be called before encode")
        inputs = self._processor(images=images, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self._model.get_image_features(**inputs)
        features = _extract_image_features(outputs)
        if features.shape[-1] != self.dim:
            # Устройство/тип берём у самих признаков, чтобы не ловить рассинхрон
            # с self.device (модель может отдать тензоры на другом устройстве).
            projection = torch.eye(
                self.dim, features.shape[-1], device=features.device, dtype=features.dtype
            )
            features = torch.nn.functional.linear(features, projection)
        return features.detach().float().cpu()

    def encode_images(self, images: list[str]) -> Tensor:
        from PIL import Image

        pil = [Image.open(self._resolve(image)).convert("RGB") for image in images]
        if not pil:
            return torch.zeros(0, self.dim)
        return self._encode_pil(pil)

    def encode_videos(self, videos: list[str], *, max_frames: int = 8) -> Tensor:
        raise NotImplementedError(
            "ClipVisionEncoder не декодирует видео; извлеките кадры и вызовите "
            "encode_images, либо используйте внешний бэкенд с модальностью video"
        )
