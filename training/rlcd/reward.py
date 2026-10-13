"""RLCD-реварды: калибровочный reward и GRPO-нормализация преимуществ."""
from __future__ import annotations

from torch import Tensor


def reward_calibration(probs: Tensor, outcomes: Tensor) -> Tensor:
    """Каждая калибровочная дельта: -|stated_probability - actual_outcome|."""
    return -(probs - outcomes).abs()


def group_advantages(rewards: Tensor, eps: float = 1e-6) -> Tensor:
    """GRPO-нормализация: стандартизация внутри группы (rewards - mean) / std."""
    mean = rewards.mean()
    std = rewards.std()
    return (rewards - mean) / (std + eps)