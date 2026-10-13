"""Энкодер состояния на базе gemma: state-строки -> батч признаков.

Загружает трансформер как *энкодер состояния*, а не генератор текста:
без цикла декодирования; наружу отдаются только агрегированные признаки.
"""
from __future__ import annotations

import torch
from torch import Tensor

from src.encoder.base import StateEncoder


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

        Padding-позиции (добавляются при ``padding=True`` до самого длинного
        элемента батча) исключаются из среднего. Иначе один и тот же state в
        разных батчах давал бы разные признаки, что ломает детерминизм и кэш
        признаков. Если маски нет (нестандартный токенизатор) — обычное среднее.
        """
        if attention_mask is None:
            return hidden.mean(dim=1)
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1.0)
        return pooled / counts

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
        if pooled.shape[-1] != self.feature_dim:
            # Фиксированная (не обучаемая) проекция: обрезка или дополнение нулями.
            # Матрицу берём на устройстве и в типе самих признаков, а не self.device,
            # иначе на CUDA-машине с CPU-моделью получаем рассинхрон устройств.
            projection = torch.eye(
                self.feature_dim, pooled.shape[-1], device=pooled.device, dtype=pooled.dtype
            )
            pooled = torch.nn.functional.linear(pooled, projection)
        return pooled.detach().float().cpu()