"""Энкодер состояния на базе encoder-only трансформера (BERT-like).

В отличие от ``GemmaStateEncoder`` (causal LM через ``AutoModelForCausalLM``),
здесь используется *двунаправленный* ``AutoModel``: последнее скрытое состояние
агрегируется masked mean-pooling с учётом ``attention_mask``. Это честный
трансформерный энкодер состояний — без цикла декодирования и без генерации.

Пример:
    encoder = TransformerStateEncoder("bert-base-uncased", feature_dim=768).load()
    features = encoder.encode(["state one", "state two"])
"""
from __future__ import annotations

import torch
from torch import Tensor

from src.encoder.base import StateEncoder
from src.encoder.pooling import masked_mean_pool, project_features


class TransformerStateEncoder(StateEncoder):
    def __init__(
        self,
        model_id: str = "bert-base-uncased",
        feature_dim: int = 768,
        dtype: torch.dtype = torch.float32,
        device: str | None = None,
        max_length: int | None = None,
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

    def load(self) -> TransformerStateEncoder:
        from transformers import AutoModel, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        if self._tokenizer.pad_token is None:
            # У encoder-only моделей (BERT) pad-токен обычно есть; для прочих
            # переиспользуем eos, чтобы padding не падал.
            self._tokenizer.pad_token = self._tokenizer.eos_token
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id
        self._model = AutoModel.from_pretrained(
            self.model_id, dtype=self.dtype, device_map=self.device
        )
        self._model.eval()
        return self

    def encode(self, states: list[str]) -> Tensor:
        """Кодирует состояния батчами, чтобы не держать в память весь набор."""
        if self._model is None:
            raise RuntimeError(
                "TransformerStateEncoder.load() must be called before encode()"
            )
        if not states:
            return torch.zeros(0, self.feature_dim)
        chunks = [
            self._encode_chunk(states[start : start + self.batch_size])
            for start in range(0, len(states), self.batch_size)
        ]
        return torch.cat(chunks, dim=0)

    @staticmethod
    def _hidden_state(output) -> Tensor:
        """Последнее скрытое состояние из вывода ``AutoModel``.

        Новые версии transformers отдают ``last_hidden_state``; некоторые
        конфигурации — только ``hidden_states[-1]``. Поддерживаем оба.
        """
        last = getattr(output, "last_hidden_state", None)
        if last is not None:
            return last
        hidden_states = getattr(output, "hidden_states", None)
        if hidden_states:
            return hidden_states[-1]
        raise RuntimeError("model output has neither last_hidden_state nor hidden_states")

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
            hidden = self._hidden_state(self._model(**inputs))
        pooled = masked_mean_pool(hidden, inputs.get("attention_mask"))
        return project_features(pooled, self.feature_dim).detach().float().cpu()
