"""Протокол энкодера состояния: state-строки -> батч признаков."""
from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from torch import Tensor


@runtime_checkable
class StateEncoder(Protocol):
    def encode(self, states: list[str]) -> Tensor:
        ...


@runtime_checkable
class TrainableStateEncoder(Protocol):
    """Энкодер, который умеет отдавать признаки **с градиентом**.

    Отдельный протокол, а не расширение ``StateEncoder``: обучать нечего хэш-
    энкодеру, и требовать от него эти методы нельзя. Дообучение идёт через тот
    же путь предобработки, что и инференс (``_encode_chunk``), — иначе замер
    дообученного энкодера не соответствовал бы его обучению.
    """

    def encode(self, states: list[str]) -> Tensor:
        ...

    def encode_trainable(self, states: list[str]) -> Tensor:
        """Признаки на устройстве модели и с сохранённым графом вычислений."""
        ...

    def set_trainable(self, flag: bool = True) -> None:
        """Включает/выключает дообучение весов (``requires_grad`` + train/eval)."""
        ...

    def trainable_parameters(self) -> list[Tensor]:
        """Параметры, попадающие в оптимизатор дообучения."""
        ...

    def save_pretrained(self, path: str | Path) -> Path:
        """Сохраняет веса и токенизатор как каталог для ``from_pretrained``."""
        ...