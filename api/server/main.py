"""System One Decision API.

Принимает state + вопросы, возвращает типизированные решения
с калиброванными вероятностями и schema_version.
"""
from __future__ import annotations

import os
from typing import Annotated, Literal

import torch
from fastapi import FastAPI
from pydantic import BaseModel, Field

from api.server.service import DecisionService
from src.sampler.questions import ChoiceSpec, NoulSpec, QuestionSpec, ScoreSpec
from src.schemas import SCHEMA_VERSION, DecisionResult


def create_service() -> DecisionService:
    """Сервис по окружению: DECISION_MODEL_ID включает реальный state-энкодер."""
    model_id = os.environ.get("DECISION_MODEL_ID")
    if not model_id:
        return DecisionService()
    from src.encoder.gemma import GemmaStateEncoder

    feature_dim = int(os.environ.get("DECISION_FEATURE_DIM", "768"))
    encoder = GemmaStateEncoder(
        model_id=model_id,
        feature_dim=feature_dim,
        dtype=torch.float32,
        device="cpu",
    )
    encoder.load()
    return DecisionService(encoder=encoder, dim=feature_dim)


app = FastAPI(title="System One Decision API", version=SCHEMA_VERSION)

service = create_service()


class NoulQuestion(BaseModel):
    type: Literal["noul"]
    instructions: str


class ChoiceQuestion(BaseModel):
    type: Literal["choice"]
    instructions: str
    options: list[str] = Field(..., min_length=2, max_length=255)


class ScoreQuestion(BaseModel):
    type: Literal["score"]
    instructions: str
    levels: list[str] = Field(..., min_length=2)


Question = NoulQuestion | ChoiceQuestion | ScoreQuestion


class DecisionRequest(BaseModel):
    state: str = Field(..., min_length=1)
    questions: dict[str, Question] = Field(..., min_length=1)


class DecisionResponse(BaseModel):
    schema_version: str
    decisions: dict[str, Annotated[DecisionResult, Field(discriminator="type")]]


def _to_specs(questions: dict[str, Question]) -> dict[str, QuestionSpec]:
    specs = {}
    for name, q in questions.items():
        if q.type == "noul":
            specs[name] = NoulSpec(instructions=q.instructions)
        elif q.type == "choice":
            specs[name] = ChoiceSpec(instructions=q.instructions, options=tuple(q.options))
        elif q.type == "score":
            specs[name] = ScoreSpec(instructions=q.instructions, levels=tuple(q.levels))
    return specs


@app.post("/decisions", response_model=DecisionResponse)
async def create_decisions(req: DecisionRequest) -> DecisionResponse:
    """Оценивает все вопросы параллельно одним прогоном энкодера + голов."""
    decisions = service.decide(req.state, _to_specs(req.questions))
    return DecisionResponse(schema_version=SCHEMA_VERSION, decisions=decisions)