"""Кэш признаков по тексту: один и тот же state кодируется многократно.

Нужен рецептам, которые несколько раз оценивают одни и те же строки (до и после
дообучения, новый и старый домен), а энкодер — самая дорогая часть. Кэш не
меняет результат: ``encode`` возвращает те же векторы, что и базовый энкодер,
просто без повторного прогона модели.
"""

from __future__ import annotations

import torch


class CachedEncoder:
    """Обёртка энкодера с кэшем по тексту промпта."""

    def __init__(self, base):
        self.base = base
        self.cache: dict[str, torch.Tensor] = {}

    def encode(self, states: list[str]) -> torch.Tensor:
        missing = [state for state in dict.fromkeys(states) if state not in self.cache]
        if missing:
            vectors = self.base.encode(missing)
            for state, vector in zip(missing, vectors, strict=True):
                self.cache[state] = vector
        return torch.stack([self.cache[state] for state in states])

    def __len__(self) -> int:
        return len(self.cache)
