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


def _read_source(ref: MediaRef) -> bytes:
    """Читает байты из файла, если путь существует, иначе трактует ref как данные."""
    if isinstance(ref, bytes):
        return ref
    try:
        path = Path(ref)
        if path.is_file():
            with path.open("rb") as handle:
                return handle.read(_MAX_BYTES)
    except (OSError, ValueError):
        pass
    return ref.encode("utf-8")


class HashVisionEncoder(VisionEncoder):
    """Детерминированный fallback: хэш содержимого в вектор признаков.

    Не требует моделей и работает офлайн — годится для тестов и для сервисов,
    где достаточно «поймать» различие входов. Для реального качества
    подключайте ``ClipVisionEncoder``.
    """

    modalities = frozenset({"image", "video"})

    def __init__(self, dim: int = VISION_FEATURE_DIM, max_bytes: int = _MAX_BYTES):
        self.dim = dim
        self.max_bytes = max_bytes

    def _vector(self, data: bytes) -> Tensor:
        vector = torch.zeros(self.dim)
        for start in range(0, len(data), 64):
            chunk = data[start : start + 64]
            digest = hashlib.blake2b(chunk, digest_size=8).digest()
            vector[int.from_bytes(digest, "big") % self.dim] += 1.0
        norm = vector.norm()
        return vector / norm if norm > 0 else vector

    def encode_images(self, images: list[MediaRef]) -> Tensor:
        rows = [self._vector(_read_source(image)[: self.max_bytes]) for image in images]
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
            data = _read_source(video)[: self.max_bytes]
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


class ClipVisionEncoder(VisionEncoder):  # pragma: no cover - требует загрузки модели
    """Реальный vision-энкодер на CLIP/SigLIP (transformers, опционально).

    Поддерживает только изображения: видео требует отдельного декодирования
    кадров (см. ``encode_images``).
    """

    modalities = frozenset({"image"})

    def __init__(
        self,
        model_id: str = "openai/clip-vit-base-patch32",
        dim: int = 512,
        device: str | None = None,
    ):
        self.model_id = model_id
        self.dim = dim
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None
        self._processor = None

    def load(self) -> ClipVisionEncoder:
        from transformers import CLIPModel, CLIPProcessor

        self._processor = CLIPProcessor.from_pretrained(self.model_id)
        self._model = CLIPModel.from_pretrained(self.model_id).to(self.device)
        self._model.eval()
        return self

    def _encode_pil(self, images: list[object]) -> Tensor:
        if self._model is None:
            raise RuntimeError("ClipVisionEncoder.load() must be called before encode")
        inputs = self._processor(images=images, return_tensors="pt").to(self.device)
        with torch.no_grad():
            features = self._model.get_image_features(**inputs)
        if features.shape[-1] != self.dim:
            features = torch.nn.functional.linear(
                features,
                torch.eye(self.dim, features.shape[-1], device=self.device),
            )
        return features.detach().float().cpu()

    def encode_images(self, images: list[str]) -> Tensor:
        from PIL import Image

        pil = [Image.open(image).convert("RGB") for image in images]
        if not pil:
            return torch.zeros(0, self.dim)
        return self._encode_pil(pil)

    def encode_videos(self, videos: list[str], *, max_frames: int = 8) -> Tensor:
        raise NotImplementedError(
            "ClipVisionEncoder не декодирует видео; извлеките кадры и вызовите "
            "encode_images, либо используйте внешний бэкенд с модальностью video"
        )
