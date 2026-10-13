"""Спецификации вопросов для голов решений."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class NoulSpec:
    instructions: str
    criteria: tuple[str, str] | None = None


@dataclass(frozen=True)
class ChoiceSpec:
    instructions: str
    options: tuple[str, ...]
    criteria: dict[str, str] | None = None


@dataclass(frozen=True)
class ScoreSpec:
    instructions: str
    levels: tuple[str, ...]
    criteria: tuple[str, ...] | None = None


QuestionSpec = NoulSpec | ChoiceSpec | ScoreSpec


def spec_to_dict(spec: QuestionSpec) -> dict[str, Any]:
    """Сериализуемое представление спецификации (для чекпоинтов и подписи)."""
    if isinstance(spec, NoulSpec):
        return {
            "type": "noul",
            "instructions": spec.instructions,
            "criteria": list(spec.criteria) if spec.criteria else None,
        }
    if isinstance(spec, ChoiceSpec):
        return {
            "type": "choice",
            "instructions": spec.instructions,
            "options": list(spec.options),
            "criteria": dict(spec.criteria) if spec.criteria else None,
        }
    if isinstance(spec, ScoreSpec):
        return {
            "type": "score",
            "instructions": spec.instructions,
            "levels": list(spec.levels),
            "criteria": list(spec.criteria) if spec.criteria else None,
        }
    raise TypeError(f"unsupported question spec: {type(spec).__name__}")


def spec_from_dict(data: dict[str, Any]) -> QuestionSpec:
    kind = data["type"]
    criteria = data.get("criteria")
    if kind == "noul":
        return NoulSpec(
            instructions=data["instructions"],
            criteria=tuple(criteria) if criteria else None,
        )
    if kind == "choice":
        return ChoiceSpec(
            instructions=data["instructions"],
            options=tuple(data["options"]),
            criteria=dict(criteria) if criteria else None,
        )
    if kind == "score":
        return ScoreSpec(
            instructions=data["instructions"],
            levels=tuple(data["levels"]),
            criteria=tuple(criteria) if criteria else None,
        )
    raise ValueError(f"unsupported question type: {kind!r}")


def questions_signature(questions: dict[str, QuestionSpec]) -> str:
    """Устойчивая подпись набора вопросов: порядок ключей не важен, порядок опций — важен."""
    payload = {name: spec_to_dict(spec) for name, spec in questions.items()}
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def build_prompt(state: str, spec: QuestionSpec) -> str:
    """Собирает вход энкодера: состояние + текст вопроса + критерии выбора."""
    parts = [f"State: {state}", f"Question: {spec.instructions}"]
    if isinstance(spec, NoulSpec):
        if spec.criteria:
            true_desc, false_desc = spec.criteria
            parts.append(f"True means: {true_desc}. False means: {false_desc}.")
    elif isinstance(spec, ChoiceSpec):
        if spec.criteria:
            rendered = "; ".join(
                f"{opt}: {spec.criteria.get(opt, '')}".rstrip(": ")
                for opt in spec.options
            )
        else:
            rendered = ", ".join(spec.options)
        parts.append(f"Options: {rendered}")
    elif isinstance(spec, ScoreSpec):
        if spec.criteria:
            rendered = "; ".join(
                f"{lvl}: {desc}" for lvl, desc in zip(spec.levels, spec.criteria)
            )
        else:
            rendered = ", ".join(spec.levels)
        parts.append(f"Scale: {rendered}")
    return "\n".join(parts)
