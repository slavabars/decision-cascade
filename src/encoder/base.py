"""Протокол энкодера состояния: state-строки -> батч признаков."""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from torch import Tensor


@runtime_checkable
class StateEncoder(Protocol):
    def encode(self, states: list[str]) -> Tensor:
        ...