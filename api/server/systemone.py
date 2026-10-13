"""Схемы и конвертеры протокола Jev/SystemOne (``POST /v1/systemone``).

Совместимо с тем, как Clef/Clef-Flash и другие decision-модели описывают
вопросы: ``instructions`` необязателен, а для ``choice``/``score`` опции можно
задавать только через ``criteria`` (ключи/элементы задают допустимые варианты).
"""
from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field

from src.sampler.questions import ChoiceSpec, NoulSpec, QuestionSpec, ScoreSpec
from src.schemas import DecisionResult


class SystemOneNoulQuestion(BaseModel):
    type: Literal["noul"]
    instructions: str | None = None
    criteria: list[str] | None = None


class SystemOneChoiceQuestion(BaseModel):
    type: Literal["choice"]
    instructions: str | None = None
    options: list[str] | None = None
    criteria: dict[str, str] | None = None


class SystemOneScoreQuestion(BaseModel):
    type: Literal["score"]
    instructions: str | None = None
    levels: list[str] | None = None
    criteria: list[str] | None = None


SystemOneQuestion = Annotated[
    SystemOneNoulQuestion | SystemOneChoiceQuestion | SystemOneScoreQuestion,
    Field(discriminator="type"),
]


class SystemOneRequest(BaseModel):
    model: str | None = None
    state: str = Field(..., min_length=1)
    questions: dict[str, SystemOneQuestion] = Field(..., min_length=1)
    images: list[Any] | None = None
    videos: list[Any] | None = None
    media_kwargs: dict[str, Any] | None = None


class SystemOneNoulAnswer(BaseModel):
    type: Literal["noul"] = "noul"
    probability: float = Field(..., ge=0.0, le=1.0)
    confidence: float = Field(..., ge=0.0, le=1.0)


class SystemOneChoiceAnswer(BaseModel):
    type: Literal["choice"] = "choice"
    choice: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    probabilities: dict[str, float]


class SystemOneScoreAnswer(BaseModel):
    type: Literal["score"] = "score"
    score: float
    confidence: float = Field(..., ge=0.0, le=1.0)
    probabilities: dict[str, float]
    legend: list[str]


SystemOneAnswer = Annotated[
    SystemOneNoulAnswer | SystemOneChoiceAnswer | SystemOneScoreAnswer,
    Field(discriminator="type"),
]


class SystemOneResponse(BaseModel):
    model: str
    answers: dict[str, SystemOneAnswer]
    usage: dict[str, Any]


def systemone_to_specs(
    questions: dict[str, SystemOneQuestion],
) -> dict[str, QuestionSpec]:
    """Вопросы протокола -> внутренние спецификации; опции выводятся из criteria."""
    specs: dict[str, QuestionSpec] = {}
    for name, question in questions.items():
        instructions = question.instructions or name
        if isinstance(question, SystemOneNoulQuestion):
            criteria = tuple(question.criteria) if question.criteria else None
            if criteria is not None and len(criteria) != 2:
                raise ValueError(f"noul question {name!r} criteria must have 2 items")
            specs[name] = NoulSpec(instructions=instructions, criteria=criteria)
        elif isinstance(question, SystemOneChoiceQuestion):
            options = (
                tuple(question.options)
                if question.options
                else tuple(question.criteria or ())
            )
            if len(options) < 2:
                raise ValueError(
                    f"choice question {name!r} needs at least 2 options or criteria"
                )
            criteria = dict(question.criteria) if question.criteria else None
            specs[name] = ChoiceSpec(
                instructions=instructions, options=options, criteria=criteria
            )
        else:
            levels = (
                tuple(question.levels)
                if question.levels
                else tuple(question.criteria or ())
            )
            if len(levels) < 2:
                raise ValueError(
                    f"score question {name!r} needs at least 2 levels or criteria"
                )
            criteria = tuple(question.criteria) if question.criteria else None
            specs[name] = ScoreSpec(
                instructions=instructions, levels=levels, criteria=criteria
            )
    return specs


def result_to_answer_model(result: DecisionResult) -> SystemOneAnswer:
    """Внутреннее решение -> pydantic-модель ответа протокола."""
    if result.type == "noul":
        return SystemOneNoulAnswer(
            probability=result.noul, confidence=result.confidence
        )
    if result.type == "choice":
        return SystemOneChoiceAnswer(
            choice=result.choice,
            confidence=result.confidence,
            probabilities=dict(result.probabilities or {}),
        )
    return SystemOneScoreAnswer(
        score=result.score,
        confidence=result.confidence,
        probabilities=dict(result.probabilities or {}),
        legend=list(result.legend),
    )
