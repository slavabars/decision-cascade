"""Сервис принятия решений: state + вопросы -> типизированные ответы.

Головы кэшируются по подписи набора вопросов и переиспользуются между
запросами; веса могут подгружаться из чекпоинтов. Кэш признаков избавляет
от повторного прогона энкодера по неизменному состоянию.
"""
from __future__ import annotations

import hashlib
import json
import logging
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from src.decision_head.heads import (
    CHOICE_HEADS,
    DecisionHeads,
    build_option_embeddings,
)
from src.encoder.base import StateEncoder
from src.encoder.vision import VisionEncoder
from src.errors import UntrainedDecisionError
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

# Политика поведения на необученных головах:
#   allow    — отдаём решения (в ответе видно ``trained: false``);
#   refuse   — отклоняем запрос (API отвечает 409 UNTRAINED_HEADS);
#   fallback — сигналим каскаду эскалировать во внешнюю модель (без каскада — как refuse).
UNTRAINED_POLICIES = ("allow", "refuse", "fallback")


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
        untrained_policy: str = "allow",
        choice_head: str = "linear",
    ):
        self.encoder = encoder or HashStateEncoder(dim)
        self.dim = dim
        self.seed = seed
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.cache_size = max(0, cache_size)
        self.vision_encoder = vision_encoder
        policy = str(untrained_policy).strip().lower()
        if policy not in UNTRAINED_POLICIES:
            raise ValueError(
                f"untrained_policy must be one of {UNTRAINED_POLICIES}, "
                f"got {untrained_policy!r}"
            )
        self.untrained_policy = policy
        mode = str(choice_head).strip().lower()
        if mode not in CHOICE_HEADS:
            raise ValueError(
                f"choice_head must be one of {CHOICE_HEADS}, got {choice_head!r}"
            )
        self.choice_head = mode
        self._heads_cache: dict[str, DecisionHeads] = {}
        self._trained: dict[str, bool] = {}
        #: Откуда загружены головы (путь к паку) — для ``/info`` и диагностики.
        self._sources: dict[str, str] = {}
        self._feature_cache: OrderedDict[str, Tensor] = OrderedDict()
        #: Был ли выполнен прогрев чекпоинтов (нужно ``readiness``: отличить
        #: «голов нет» от «головы не читаются»).
        self._warmup_done = False
        # Эндпоинты синхронные и обслуживаются threadpool'ом FastAPI, поэтому
        # общие кэши (головы и признаков) прикрываем блокировкой.
        self._lock = threading.RLock()

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

    @staticmethod
    def _heads_key(dim: int, questions: dict[str, QuestionSpec]) -> str:
        return f"{dim}:{questions_signature(questions)}"

    def _pack_dirs(self) -> list[Path]:
        """Каталоги паков: ``<checkpoint_dir>/*/metadata.json`` + ``heads.pt``.

        Многодоменная раскладка — по каталогу на пак
        (``checkpoints/packs/<домен>/``); плоские ``*.pt`` в корне остаются
        поддержанными для старых установок.
        """
        if self.checkpoint_dir is None or not self.checkpoint_dir.is_dir():
            return []
        return [
            metadata.parent
            for metadata in sorted(self.checkpoint_dir.glob("*/metadata.json"))
            if (metadata.parent / "heads.pt").is_file()
        ]

    def _checkpoint_paths(self) -> list[Path]:
        """Все читаемые чекпоинты: корневые ``*.pt`` и паки в подкаталогах."""
        if self.checkpoint_dir is None or not self.checkpoint_dir.is_dir():
            return []
        paths = sorted(self.checkpoint_dir.glob("*.pt"))
        paths.extend(pack / "heads.pt" for pack in self._pack_dirs())
        return paths

    def _find_checkpoint(
        self, questions: dict[str, QuestionSpec], dim: int | None = None
    ) -> Path | None:
        """Ищет пак под подпись вопросов в корне и в подкаталогах.

        Порядок: ``<dir>/<signature>.pt`` (старая раскладка), затем пак, чей
        ``metadata.json`` объявляет эту подпись, затем ``*/<signature>.pt``.
        Если задан ``dim``, паки с другим ``feature_dim`` пропускаются: один и
        тот же набор вопросов может быть обучен под разные энкодеры, и сервис
        должен взять совместимый, а не первый по алфавиту.
        """
        if self.checkpoint_dir is None:
            return None
        signature = questions_signature(questions)
        direct = self.checkpoint_dir / f"{signature}.pt"
        if direct.is_file():
            return direct
        fallback: Path | None = None
        for pack in self._pack_dirs():
            metadata = pack / "metadata.json"
            try:
                payload = json.loads(metadata.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                logger.warning("cannot read pack metadata %s", metadata)
                continue
            if payload.get("questions_signature") != signature:
                continue
            pack_dim = payload.get("feature_dim")
            if dim is not None and pack_dim is not None and int(pack_dim) != dim:
                fallback = fallback or pack / "heads.pt"
                continue
            return pack / "heads.pt"
        if fallback is not None:
            # Совместимого по dim пака нет — вернём хоть какой-то: загрузчик
            # проверит dim по самому чекпоинту и честно предупредит.
            return fallback
        nested = sorted(self.checkpoint_dir.glob(f"*/{signature}.pt"))
        return nested[0] if nested else None


    def _ensure_heads(
        self, dim: int, questions: dict[str, QuestionSpec]
    ) -> tuple[DecisionHeads, bool]:
        """Возвращает головы и признак «обучены» (загружен обученный чекпоинт)."""
        key = self._heads_key(dim, questions)
        with self._lock:
            heads = self._heads_cache.get(key)
            if heads is None:
                option_embeddings = None
                if self.choice_head == "option_scoring":
                    option_embeddings = build_option_embeddings(
                        questions, self._encode_prompts
                    )
                heads = DecisionHeads(
                    dim,
                    questions,
                    seed=self.seed,
                    choice_head=self.choice_head,
                    option_embeddings=option_embeddings,
                )
                loaded = self._load_checkpoint_into(heads, dim)
                self._heads_cache[key] = heads
                self._trained[key] = loaded is not None
                if loaded is not None:
                    self._sources[key] = str(loaded)
                trained = self._trained[key]
            else:
                trained = self._trained.get(key, False)
        return heads, trained

    def get_heads(self, dim: int, questions: dict[str, QuestionSpec]) -> DecisionHeads:
        """Возвращает головы для набора вопросов, переиспользуя их между запросами."""
        return self._ensure_heads(dim, questions)[0]

    def is_trained(self, questions: dict[str, QuestionSpec]) -> bool:
        """Есть ли под текущий набор вопросов загруженный обученный чекпоинт.

        Ключ строится по ``self.dim``: энкодеры сервиса создаются под эту
        размерность, поэтому она совпадает с фактической размерностью признаков.
        """
        key = self._heads_key(self.dim, questions)
        with self._lock:
            return self._trained.get(key, False)

    def info(self) -> dict[str, Any]:
        """Снимок конфигурации и загруженных голов — для эндпоинта ``/info``."""
        with self._lock:
            heads = [
                {
                    "dim": int(key.split(":", 1)[0]),
                    "signature": key.split(":", 1)[1],
                    "trained": self._trained.get(key, False),
                    "source": self._sources.get(key),
                }
                for key in self._heads_cache
            ]
        return {
            "encoder": type(self.encoder).__name__,
            "dim": self.dim,
            "untrained_policy": self.untrained_policy,
            "choice_head": self.choice_head,
            "checkpoint_dir": (
                str(self.checkpoint_dir) if self.checkpoint_dir else None
            ),
            "heads": heads,
        }

    @staticmethod
    def _component_loaded(component: Any) -> bool:
        """Загружен ли компонент: у трансформеров признак — непустой ``_model``."""
        if hasattr(component, "_model"):
            return component._model is not None
        loaded = getattr(component, "loaded", None)
        if isinstance(loaded, bool):
            return loaded
        return True

    def readiness(self) -> dict[str, Any]:
        """Готовность сервиса: энкодер загружен, а головы есть и читаются.

        Политика ``refuse`` требует хотя бы одного чекпоинта: иначе первый же
        запрос получит 409, и балансировщику незачем считать инстанс готовым.
        При ``allow``/``fallback`` сервис отвечает и без паков (демо-режим или
        эскалация во внешнюю модель), поэтому отсутствие чекпоинтов не делает
        его неготовым.
        """
        reasons: list[str] = []
        encoder_ready = self._component_loaded(self.encoder)
        if not encoder_ready:
            reasons.append("encoder is not loaded")
        vision_ready = True
        if self.vision_encoder is not None:
            vision_ready = self._component_loaded(self.vision_encoder)
            if not vision_ready:
                reasons.append("vision encoder is not loaded")
        packs = self._checkpoint_paths()
        with self._lock:
            heads_loaded = sum(1 for trained in self._trained.values() if trained)
        if self.untrained_policy == "refuse":
            if not packs:
                reasons.append("no trained checkpoints for untrained_policy=refuse")
            elif self._warmup_done and heads_loaded == 0:
                reasons.append("checkpoints are not readable by this service")
        return {
            "ready": not reasons,
            "reasons": reasons,
            "encoder": type(self.encoder).__name__,
            "encoder_loaded": encoder_ready,
            "vision_loaded": vision_ready,
            "untrained_policy": self.untrained_policy,
            "checkpoint_dir": (
                str(self.checkpoint_dir) if self.checkpoint_dir else None
            ),
            "checkpoints": len(packs),
            "heads_loaded": heads_loaded,
        }

    def warmup(self) -> int:
        """Загружает головы из чекпоинтов в кэш при старте.

        Обходит корневые ``*.pt`` и паки ``*/heads.pt`` с ``metadata.json`` —
        так один сервис может держать несколько доменов и выбирать пак по
        подписи вопросов. Прогрев делает ``/ready`` правдивым (головы
        действительно загружены) и убирает задержку первого запроса. Паки другой
        размерности или другого режима ``choice_head`` пропускаются: они всё
        равно не подойдут этому сервису. Ошибка чтения одного файла не мешает
        остальным — она видна в логе и в ``readiness``.
        """
        self._warmup_done = True
        loaded = 0
        for path in self._checkpoint_paths():
            try:
                heads = DecisionHeads.load_checkpoint(path, seed=self.seed)
            except Exception as exc:  # noqa: BLE001 - один битый пак не блокирует старт
                logger.warning(
                    "cannot load checkpoint %s: %s", path.name, type(exc).__name__
                )
                continue
            if heads.dim != self.dim:
                logger.info(
                    "skip checkpoint %s: dim=%s, service dim=%s",
                    path.name,
                    heads.dim,
                    self.dim,
                )
                continue
            if heads.choice_head != self.choice_head:
                logger.warning(
                    "skip checkpoint %s: choice_head=%s, service choice_head=%s",
                    path.name,
                    heads.choice_head,
                    self.choice_head,
                )
                continue
            key = self._heads_key(heads.dim, heads.questions)
            with self._lock:
                if key in self._heads_cache:
                    # ``heads.pt`` и ``<signature>.pt`` — один и тот же пак.
                    previous = self._sources.get(key)
                    if previous is not None and previous != str(path):
                        logger.warning(
                            "skip checkpoint %s: questions signature %s is "
                            "already served by %s",
                            path,
                            key.split(":", 1)[1],
                            previous,
                        )
                    continue
                self._heads_cache[key] = heads
                self._trained[key] = True
                self._sources[key] = str(path)
            loaded += 1
        return loaded

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

    def _load_checkpoint_into(self, heads: DecisionHeads, dim: int) -> Path | None:
        path = self._find_checkpoint(heads.questions, dim)
        if path is None:
            # Молчаливое обслуживание необученными головами — источник трудно
            # диагностируемых низких метрик. Если каталог есть, а нужной подписи
            # нет (например, критерии в вопросе не совпали с обучением) — предупреждаем.
            if self.checkpoint_dir is not None and self.checkpoint_dir.exists():
                logger.warning(
                    "no checkpoint for questions signature %s in %s; "
                    "serving untrained heads",
                    questions_signature(heads.questions),
                    self.checkpoint_dir,
                )
            else:
                logger.info(
                    "no checkpoint dir %s; serving untrained heads",
                    self.checkpoint_dir,
                )
            return None
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if int(payload.get("dim", -1)) != dim:
            logger.warning(
                "checkpoint dim=%s does not match features dim=%s; ignoring %s",
                payload.get("dim"),
                dim,
                path,
            )
            return None
        checkpoint_choice_head = str(payload.get("choice_head", "linear"))
        if checkpoint_choice_head != heads.choice_head:
            logger.warning(
                "checkpoint choice_head=%s does not match service choice_head=%s; "
                "ignoring %s",
                checkpoint_choice_head,
                heads.choice_head,
                path,
            )
            return None
        heads.load_state_dict(payload["state_dict"])
        heads.set_temperatures(payload.get("temperature") or {})
        return path

    def _check_trained(
        self, questions: dict[str, QuestionSpec], trained: bool
    ) -> None:
        """Не отдаём решения необученными головами, если политика это запрещает."""
        if trained or self.untrained_policy == "allow":
            return
        raise UntrainedDecisionError(
            questions_signature(questions),
            str(self.checkpoint_dir) if self.checkpoint_dir else None,
        )

    def _encode_prompts(self, prompts: list[str], *, use_cache: bool = True) -> Tensor:
        if not use_cache or not self.cache_size:
            return self.encoder.encode(prompts)
        # Блокировка оберегает только кэш (OrderedDict). Сам forward энкодера
        # выполняется вне блокировки, иначе синхронные эндпоинты в threadpool
        # сериализовались бы на время дорогого encode/GPU-прохода.
        with self._lock:
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
            with self._lock:
                for j, i in enumerate(missing_idx):
                    vector = features[j]
                    rows[i] = vector
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
        """Решает набор состояний за один вызов энкодера (промпты — батчами)."""
        if not states or not questions:
            return [{} for _ in states]
        names = list(questions)
        prompts = [
            build_prompt(state, questions[name]) for state in states for name in names
        ]
        features = self._encode_prompts(prompts)
        heads, trained = self._ensure_heads(features.shape[-1], questions)
        self._check_trained(questions, trained)
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
        heads, trained = self._ensure_heads(features.shape[-1], questions)
        self._check_trained(questions, trained)
        results = sample_decisions(heads, features)
        for name, result in results.items():
            dist = getattr(result, "probabilities", None) or {"p": result.noul}
            # Логируем только решение (вопрос, тип, распределение, confidence).
            # state, images и тела /v1/feedback сюда не попадают — это
            # зафиксировано тестом tests/test_logging_privacy.py.
            logger.info(
                "decision name=%s type=%s distribution=%s confidence=%.3f",
                name,
                result.type,
                dist,
                result.confidence,
            )
        return results
