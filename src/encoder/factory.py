"""Фабрика текстовых энкодеров состояния.

Единая точка выбора реализации: causal LM (``GemmaStateEncoder``) или
encoder-only трансформер (``TransformerStateEncoder``). Вид задаётся явно
параметром ``kind`` или переменной окружения ``DECISION_ENCODER_KIND``
(``causal`` по умолчанию, ``encoder`` — для BERT-like моделей).
"""
from __future__ import annotations

import os

import torch

ENCODER_KINDS = ("causal", "encoder")


def create_state_encoder(
    model_id: str,
    feature_dim: int,
    *,
    kind: str | None = None,
    dtype: torch.dtype = torch.float32,
    device: str | None = None,
    max_length: int | None = 256,
    batch_size: int = 16,
):
    """Создаёт (но не загружает) энкодер заданного вида.

    ``kind=None`` читает ``DECISION_ENCODER_KIND``. Неизвестный вид — ошибка,
    чтобы опечатка не приводила молча к causal-энкодеру.
    """
    resolved = (kind or os.environ.get("DECISION_ENCODER_KIND", "causal")).strip().lower()
    if resolved not in ENCODER_KINDS:
        raise ValueError(
            f"unknown encoder kind {resolved!r}; expected one of {ENCODER_KINDS}"
        )
    if resolved == "encoder":
        from src.encoder.transformer import TransformerStateEncoder

        return TransformerStateEncoder(
            model_id=model_id,
            feature_dim=feature_dim,
            dtype=dtype,
            device=device,
            max_length=max_length,
            batch_size=batch_size,
        )
    from src.encoder.gemma import GemmaStateEncoder

    return GemmaStateEncoder(
        model_id=model_id,
        feature_dim=feature_dim,
        dtype=dtype,
        device=device,
        max_length=max_length,
        batch_size=batch_size,
    )
