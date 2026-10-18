"""Энкодер состояния на базе gemma: state-строки -> батч признаков.

Загружает трансформер как *энкодер состояния*, а не генератор текста:
без цикла декодирования; наружу отдаются только агрегированные признаки.
Этот же путь предобработки используется для дообучения энкодера
(``encode_trainable``), чтобы замер шёл ровно на том, чему энкодер обучен.
"""
from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path

import torch
from torch import Tensor

from src.encoder.base import StateEncoder
from src.encoder.pooling import masked_mean_pool, project_features


class GemmaStateEncoder(StateEncoder):
    def __init__(
        self,
        model_id: str = "google/gemma-4-12B",
        feature_dim: int = 768,
        dtype: torch.dtype = torch.float32,
        device: str | None = None,
        max_length: int | None = 256,
        batch_size: int = 16,
    ):
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        self.model_id = model_id
        self.feature_dim = feature_dim
        self.dtype = dtype
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.batch_size = batch_size
        self._model = None
        self._tokenizer = None

    def load(self) -> GemmaStateEncoder:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id, dtype=self.dtype, device_map=self.device
        )
        self._model.eval()
        return self

    def encode(self, states: list[str]) -> Tensor:
        """Кодирует состояния батчами, чтобы не держать в память весь набор."""
        if self._model is None:
            raise RuntimeError("GemmaStateEncoder.load() must be called before encode()")
        if not states:
            return torch.zeros(0, self.feature_dim)
        chunks = [
            self._encode_chunk(states[start : start + self.batch_size])
            for start in range(0, len(states), self.batch_size)
        ]
        return torch.cat(chunks, dim=0)

    @staticmethod
    def _mean_pool(hidden: Tensor, attention_mask: Tensor | None) -> Tensor:
        """Mean-pooling по токенам состояния с учётом ``attention_mask``.

        Тонкая обёртка над общим ``masked_mean_pool`` (общая логика вынесена в
        ``src.encoder.pooling``, чтобы её переиспользовал encoder-only энкодер).
        """
        return masked_mean_pool(hidden, attention_mask)

    def _encode_chunk(self, states: list[str], *, trainable: bool = False) -> Tensor:
        """Признаки одного чанка.

        ``trainable=True`` оставляет граф вычислений и устройство модели — это
        путь дообучения (``encode_trainable``). ``False`` — путь инференса: без
        графа, в float32 и на CPU, чтобы признаки кэшировались и не держали
        память GPU.
        """
        inputs = self._tokenizer(
            states,
            return_tensors="pt",
            padding=True,
            truncation=self.max_length is not None,
            max_length=self.max_length,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        context = nullcontext() if trainable else torch.no_grad()
        with context:
            hidden = self._model(**inputs, output_hidden_states=True).hidden_states[-1]
        pooled = self._mean_pool(hidden, inputs.get("attention_mask"))
        projected = project_features(pooled, self.feature_dim)
        if trainable:
            return projected.float()
        return projected.detach().float().cpu()

    def encode_trainable(self, states: list[str]) -> Tensor:
        """Кодирует состояния батчами **с градиентом** (для дообучения).

        В отличие от :meth:`encode` признаки остаются на устройстве модели и в
        графе вычислений, поэтому потеря голов течёт в веса энкодера. Для
        инференса этот путь не годится: он дороже и недетерминирован — в режиме
        ``train()`` работает dropout.
        """
        if self._model is None:
            raise RuntimeError(
                "GemmaStateEncoder.load() must be called before encode_trainable()"
            )
        if not states:
            return torch.zeros(0, self.feature_dim, device=self.device)
        chunks = [
            self._encode_chunk(states[start : start + self.batch_size], trainable=True)
            for start in range(0, len(states), self.batch_size)
        ]
        return torch.cat(chunks, dim=0)

    def set_trainable(self, flag: bool = True) -> None:
        """Включает (``True``) или выключает (``False``) дообучение весов.

        Кроме ``requires_grad`` переключается режим модуля: ``True`` — ``train()``
        (включая dropout), ``False`` — ``eval()``, как при обычном инференсе.
        """
        if self._model is None:
            raise RuntimeError(
                "GemmaStateEncoder.load() must be called before set_trainable()"
            )
        for param in self._model.parameters():
            param.requires_grad_(flag)
        self._model.train(flag)

    def trainable_parameters(self) -> list[Tensor]:
        """Параметры, попадающие в оптимизатор дообучения."""
        if self._model is None:
            raise RuntimeError(
                "GemmaStateEncoder.load() must be called before trainable_parameters()"
            )
        return [param for param in self._model.parameters() if param.requires_grad]

    def save_pretrained(self, path: str | Path) -> Path:
        """Сохраняет дообученные веса и токенизатор как каталог ``from_pretrained``.

        Отдельного формата чекпоинта энкодера нет: каталог подходит и как
        ``--encoder`` замерного харнесса, и как ``DECISION_MODEL_ID`` сервиса.
        """
        if self._model is None or self._tokenizer is None:
            raise RuntimeError(
                "GemmaStateEncoder.load() must be called before save_pretrained()"
            )
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self._model.save_pretrained(path)
        self._tokenizer.save_pretrained(path)
        return path