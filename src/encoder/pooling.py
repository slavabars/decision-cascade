"""Общие операции текстовых энкодеров: masked pooling и фиксированная проекция."""
from __future__ import annotations

import torch
from torch import Tensor


def masked_mean_pool(hidden: Tensor, attention_mask: Tensor | None) -> Tensor:
    """Mean-pooling по токенам с учётом ``attention_mask``.

    Padding-позиции (добавляются при ``padding=True`` до самого длинного
    элемента батча) исключаются из среднего. Иначе один и тот же state в
    разных батчах давал бы разные признаки, что ломает детерминизм и кэш
    признаков. Если маски нет (нестандартный токенизатор) — обычное среднее.
    Все-padding строка даёт нулевой вектор, а не NaN (``clamp(min=1)``).
    """
    if attention_mask is None:
        return hidden.mean(dim=1)
    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
    pooled = (hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1.0)
    return pooled / counts


def project_features(pooled: Tensor, feature_dim: int) -> Tensor:
    """Фиксированная (не обучаемая) проекция признаков к ``feature_dim``.

    Матрицу берём на устройстве и в типе самих признаков, а не на ``device``
    модели, иначе на CUDA-машине с CPU-моделью получим рассинхрон устройств.
    """
    if pooled.shape[-1] == feature_dim:
        return pooled
    projection = torch.eye(
        feature_dim, pooled.shape[-1], device=pooled.device, dtype=pooled.dtype
    )
    return torch.nn.functional.linear(pooled, projection)
