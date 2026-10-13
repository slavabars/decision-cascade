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

from api.server.service import DEFAULT_CHECKPOINT_DIR, DecisionService
from src.sampler.questions import ChoiceSpec, NoulSpec, QuestionSpec, ScoreSpec
from src.schemas import SCHEMA_VERSION, DecisionResult


def create_service() -> DecisionService:
    """Сервис по окружению: DECISION_MODEL_ID включает реальный state-энкодер."""
    checkpoint_dir = os.environ.get("DECISION_CHECKPOINT_DIR", DEFAULT_CHECKPOINT_DIR)
    model_id = os.environ.get("DECISION_MODEL_ID")
    if not model_id:
        return DecisionService(checkpoint_dir=checkpoint_dir)
    from src.encoder.gemma import GemmaStateEncoder

    feature_dim = int(os.environ.get("DECISION_FEATURE_DIM", "768"))
    encoder = GemmaStateEncoder(
        model_id=model_id,
        feature_dim=feature_dim,
        dtype=torch.float32,
        device="cpu",
    )
    encoder.load()
    return DecisionService(
        encoder=encoder,
        dim=feature_dim,
        checkpoint_dir=checkpoint_dir,
    )


app = FastAPI(title="System One Decision API", version=SCHEMA_VERSION)

service = create_service()


class NoulQuestion(BaseModel):
    type: Literal["noul"]
    instructions: str
    criteria: list[str] | None = Field(None, min_length=2, max_length=2)


class ChoiceQuestion(BaseModel):
    type: Literal["choice"]
    instructions: str
    options: list[str] = Field(..., min_length=2, max_length=255)
    criteria: dict[str, str] | None = None


class ScoreQuestion(BaseModel):
    type: Literal["score"]
    instructions: str
    levels: list[str] = Field(..., min_length=2)
    criteria: list[str] | None = None


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
            criteria = tuple(q.criteria) if q.criteria else None
            specs[name] = NoulSpec(instructions=q.instructions, criteria=criteria)
        elif q.type == "choice":
            criteria = dict(q.criteria) if q.criteria else None
            specs[name] = ChoiceSpec(
                instructions=q.instructions, options=tuple(q.options), criteria=criteria
            )
        elif q.type == "score":
            criteria = tuple(q.criteria) if q.criteria else None
            specs[name] = ScoreSpec(
                instructions=q.instructions, levels=tuple(q.levels), criteria=criteria
            )
    return specs


@app.post("/decisions", response_model=DecisionResponse)
async def create_decisions(req: DecisionRequest) -> DecisionResponse:
    """Оценивает все вопросы параллельно одним прогоном энкодера + голов."""
    decisions = service.decide(req.state, _to_specs(req.questions))
    return DecisionResponse(schema_version=SCHEMA_VERSION, decisions=decisions)