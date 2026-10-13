"""Спецификации вопросов для голов решений."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NoulSpec:
    instructions: str


@dataclass(frozen=True)
class ChoiceSpec:
    instructions: str
    options: tuple[str, ...]


@dataclass(frozen=True)
class ScoreSpec:
    instructions: str
    levels: tuple[str, ...]


QuestionSpec = NoulSpec | ChoiceSpec | ScoreSpec