"""Сервис принятия решений: state + вопросы -> типизированные ответы.

Головы кэшируются по подписи набора вопросов и переиспользуются между
запросами; веса могут подгружаться из чекпоинтов. Кэш признаков избавляет
от повторного прогона энкодера по неизменному состоянию.
"""
from __future__ import annotations

import hashlib
import logging
from collections import OrderedDict
from pathlib import Path

import torch
from torch import Tensor

from src.decision_head.heads import DecisionHeads
from src.encoder.base import StateEncoder
from src.encoder.vision import VisionEncoder
from src.sampler.parallel import sample_decisions
from src.sampler.questions import QuestionSpec, build_prompt, questions_signature
from src.schemas import DecisionResult

logger = logging.getLogger("decision.service")

FEATURE_DIM = 64
DEFAULT_CHECKPOINT_DIR = "checkpoints"
DEFAULT_MEDIA_WEIGHT = 0.5
TEXT_MODALITY = "text"
IMAGE_MODALITY = "image"
VIDEO_MODALITY = "video"


class HashStateEncoder:
    """Детерминированный fallback-энкодер: bag-of-токенов по хэшам."""

    def __init__(self, dim: int = FEATURE_DIM):
        self.dim = dim

    def encode(self, states: list[str]) -> Tensor:
        features = torch.zeros(len(states), self.dim)
        for row, text in enumerate(states):
            for token in text.lower().split():
                digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).hexdigest()
                features[row, int(digest, 16) % self.dim] += 1.0
            norm = features[row].norm()
            if norm > 0:
                features[row] /= norm
        return features


class DecisionService:
    def __init__(
        self,
        encoder: StateEncoder | None = None,
        dim: int = FEATURE_DIM,
        seed: int = 0,
        checkpoint_dir: str | Path | None = None,
        cache_size: int = 256,
        vision_encoder: VisionEncoder | None = None,
    ):
        self.encoder = encoder or HashStateEncoder(dim)
        self.dim = dim
        self.seed = seed
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.cache_size = max(0, cache_size)
        self.vision_encoder = vision_encoder
        self._heads_cache: dict[str, DecisionHeads] = {}
        self._feature_cache: OrderedDict[str, Tensor] = OrderedDict()

    @property
    def modalities(self) -> frozenset[str]:
        """Модальности входа, которые сервис способен обработать."""
        supported = {TEXT_MODALITY}
        if self.vision_encoder is not None:
            supported |= set(
                getattr(
                    self.vision_encoder,
                    "modalities",
                    {IMAGE_MODALITY, VIDEO_MODALITY},
                )
            )
        return frozenset(supported)

    def get_heads(self, dim: int, questions: dict[str, QuestionSpec]) -> DecisionHeads:
        """Возвращает головы для набора вопросов, переиспользуя их между запросами."""
        signature = questions_signature(questions)
        key = f"{dim}:{signature}"
        heads = self._heads_cache.get(key)
        if heads is None:
            heads = DecisionHeads(dim, questions, seed=self.seed)
            self._load_checkpoint_into(heads, dim)
            self._heads_cache[key] = heads
        return heads

    def save_heads(
        self,
        dim: int,
        questions: dict[str, QuestionSpec],
        directory: str | Path | None = None,
    ) -> Path:
        """Сохраняет текущие (в т.ч. обученные) головы в чекпоинт."""
        base = (
            Path(directory)
            if directory is not None
            else (self.checkpoint_dir or Path(DEFAULT_CHECKPOINT_DIR))
        )
        path = base / f"{questions_signature(questions)}.pt"
        return self.get_heads(dim, questions).save_checkpoint(path)

    def _checkpoint_path(self, questions: dict[str, QuestionSpec]) -> Path | None:
        if self.checkpoint_dir is None:
            return None
        return self.checkpoint_dir / f"{questions_signature(questions)}.pt"

    def _load_checkpoint_into(self, heads: DecisionHeads, dim: int) -> None:
        path = self._checkpoint_path(heads.questions)
        if path is None or not path.exists():
            return
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if int(payload.get("dim", -1)) != dim:
            logger.warning(
                "checkpoint dim=%s does not match features dim=%s; ignoring %s",
                payload.get("dim"),
                dim,
                path,
            )
            return
        heads.load_state_dict(payload["state_dict"])

    def _encode_prompts(self, prompts: list[str], *, use_cache: bool = True) -> Tensor:
        if not use_cache:
            return self.encoder.encode(prompts)
        rows: list[Tensor | None] = [None] * len(prompts)
        missing: list[str] = []
        missing_idx: list[int] = []
        for i, prompt in enumerate(prompts):
            cached = self._feature_cache.get(prompt)
            if cached is not None:
                self._feature_cache.move_to_end(prompt)
                rows[i] = cached
            else:
                missing.append(prompt)
                missing_idx.append(i)
        if missing:
            features = self.encoder.encode(missing)
            if features.shape[0] != len(missing):
                raise ValueError(
                    "encoder returned "
                    f"{features.shape[0]} rows for {len(missing)} prompts"
                )
            for j, i in enumerate(missing_idx):
                vector = features[j]
                rows[i] = vector
                if self.cache_size:
                    self._feature_cache[prompts[i]] = vector
                    self._feature_cache.move_to_end(prompts[i])
                    while len(self._feature_cache) > self.cache_size:
                        self._feature_cache.popitem(last=False)
        return torch.stack([row for row in rows if row is not None])

    def _encode_media(
        self,
        images: list[str] | None,
        videos: list[str] | None,
        media_kwargs: dict[str, object] | None,
    ) -> Tensor | None:
        """Усреднённый вектор модальностей состояния (общий для всех вопросов)."""
        kwargs = media_kwargs or {}
        vectors: list[Tensor] = []
        if images:
            vectors.append(self.vision_encoder.encode_images(images).mean(dim=0))
        if videos:
            max_frames = int(kwargs.get("max_frames", 8))
            vectors.append(
                self.vision_encoder.encode_videos(videos, max_frames=max_frames).mean(
                    dim=0
                )
            )
        if not vectors:
            return None
        return torch.stack(vectors).mean(dim=0)

    @staticmethod
    def _fuse(
        features: Tensor, media: Tensor | None, weight: float
    ) -> Tensor:
        if media is None or weight == 0.0:
            return features
        projected = media
        if projected.shape[-1] != features.shape[-1]:
            projection = torch.eye(
                features.shape[-1], projected.shape[-1]
            )
            projected = projection @ projected
        return features + (weight * projected).unsqueeze(0)

    def decide_many(
        self,
        states: list[str],
        questions: dict[str, QuestionSpec],
    ) -> list[dict[str, DecisionResult]]:
        """Решает набор состояний одним прогоном энкодера (все промпты за раз)."""
        if not states or not questions:
            return [{} for _ in states]
        names = list(questions)
        prompts = [
            build_prompt(state, questions[name]) for state in states for name in names
        ]
        features = self._encode_prompts(prompts)
        heads = self.get_heads(features.shape[-1], questions)
        width = len(names)
        return [
            sample_decisions(heads, features[i * width : (i + 1) * width])
            for i in range(len(states))
        ]

    def decide(
        self,
        state: str,
        questions: dict[str, QuestionSpec],
        *,
        images: list[str] | None = None,
        videos: list[str] | None = None,
        media_kwargs: dict[str, object] | None = None,
    ) -> dict[str, DecisionResult]:
        if not questions:
            return {}
        has_media = bool(images or videos)
        if has_media:
            supported = self.modalities
            requested = set()
            if images:
                requested.add(IMAGE_MODALITY)
            if videos:
                requested.add(VIDEO_MODALITY)
            missing = requested - supported
            if missing:
                raise ValueError(
                    f"service does not support modalities {sorted(missing)}; "
                    "configure a matching vision encoder"
                )
        prompts = [build_prompt(state, spec) for spec in questions.values()]
        features = self._encode_prompts(prompts, use_cache=not has_media)
        media = self._encode_media(images, videos, media_kwargs)
        weight = float((media_kwargs or {}).get("media_weight", DEFAULT_MEDIA_WEIGHT))
        features = self._fuse(features, media, weight)
        heads = self.get_heads(features.shape[-1], questions)
        results = sample_decisions(heads, features)
        for name, result in results.items():
            dist = getattr(result, "probabilities", None) or {"p": result.noul}
            logger.info(
                "decision name=%s type=%s distribution=%s confidence=%.3f",
                name,
                result.type,
                dist,
                result.confidence,
            )
        return results
