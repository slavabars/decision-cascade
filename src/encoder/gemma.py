"""Энкодер состояния на базе gemma: state-строки -> батч признаков.

Загружает трансформер как *энкодер состояния*, а не генератор текста:
без цикла декодирования; наружу отдаются только агрегированные признаки.
"""
from __future__ import annotations

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

    def _encode_chunk(self, states: list[str]) -> Tensor:
        inputs = self._tokenizer(
            states,
            return_tensors="pt",
            padding=True,
            truncation=self.max_length is not None,
            max_length=self.max_length,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            hidden = self._model(**inputs, output_hidden_states=True).hidden_states[-1]
        pooled = self._mean_pool(hidden, inputs.get("attention_mask"))
        return project_features(pooled, self.feature_dim).detach().float().cpu()