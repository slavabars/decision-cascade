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
    ):
        self.model_id = model_id
        self.feature_dim = feature_dim
        self.dtype = dtype
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
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
        if self._model is None:
            raise RuntimeError("GemmaStateEncoder.load() must be called before encode()")
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
        pooled = hidden.mean(dim=1)
        if pooled.shape[-1] != self.feature_dim:
            projected = torch.nn.functional.linear(
                pooled, torch.eye(self.feature_dim, pooled.shape[-1], device=self.device)
            )
            pooled = projected
        return pooled.detach().float().cpu()