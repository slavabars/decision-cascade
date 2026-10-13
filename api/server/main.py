"""decision-cascade API.

Принимает state + вопросы, возвращает типизированные решения
с калиброванными вероятностями и schema_version.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated, Literal

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator, model_validator

from api.server.service import DEFAULT_CHECKPOINT_DIR, DecisionService
from api.server.systemone import (
    SystemOneRequest,
    SystemOneResponse,
    result_to_answer_model,
    systemone_to_specs,
)
from src import __version__
from src.backends.base import Backend, missing_modalities, requested_modalities
from src.backends.cascade import CascadeBackend
from src.backends.http import HttpSystemOneBackend, parse_modalities
from src.backends.local import LocalHeadsBackend
from src.device import check_dtype_device, resolve_device, resolve_dtype
from src.encoder.vision import VisionEncoder
from src.sampler.questions import ChoiceSpec, NoulSpec, QuestionSpec, ScoreSpec
from src.schemas import SCHEMA_VERSION, DecisionResult


def create_vision_encoder() -> VisionEncoder | None:
    """Vision-энкодер по окружению: DECISION_VISION_MODEL_ID включает CLIP."""
    model_id = os.environ.get("DECISION_VISION_MODEL_ID")
    if not model_id:
        return None
    from src.encoder.vision import ClipVisionEncoder

    media_root = os.environ.get("DECISION_MEDIA_ROOT")
    encoder = ClipVisionEncoder(
        model_id=model_id,
        dim=parse_dim(
            "DECISION_VISION_DIM",
            os.environ.get("DECISION_VISION_DIM"),
            default=512,
            lo=1,
            hi=8192,
        ),
        device=resolve_device(os.environ.get("DECISION_DEVICE")),
        media_root=Path(media_root) if media_root else None,
    )
    encoder.load()
    return encoder


def parse_dim(
    name: str, raw: str | None, *, default: int, lo: int, hi: int
) -> int:
    """Разбирает размерность из окружения и держит её в разумных границах."""
    value = default if raw is None else raw
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"{name} must be an int in [{lo}, {hi}], got {value!r}"
        ) from exc
    if not lo <= parsed <= hi:
        raise RuntimeError(f"{name} must be in [{lo}, {hi}], got {parsed}")
    return parsed


def parse_max_length(raw: str) -> int | None:
    """Разбирает DECISION_MAX_LENGTH: положительный int или None (без обрезки)."""
    value = raw.strip().lower()
    if value in {"", "none", "0"}:
        return None
    try:
        parsed = int(value)
    except ValueError as exc:
        raise RuntimeError(
            f"DECISION_MAX_LENGTH must be a positive int or 'none', got {raw!r}"
        ) from exc
    return parsed if parsed >= 1 else None


def create_service() -> DecisionService:
    """Сервис по окружению: DECISION_MODEL_ID включает реальный state-энкодер."""
    checkpoint_dir = os.environ.get("DECISION_CHECKPOINT_DIR", DEFAULT_CHECKPOINT_DIR)
    vision_encoder = create_vision_encoder()
    model_id = os.environ.get("DECISION_MODEL_ID")
    if not model_id:
        return DecisionService(
            checkpoint_dir=checkpoint_dir, vision_encoder=vision_encoder
        )
    from src.encoder.factory import create_state_encoder

    feature_dim = parse_dim(
        "DECISION_FEATURE_DIM",
        os.environ.get("DECISION_FEATURE_DIM"),
        default=768,
        lo=1,
        hi=8192,
    )
    max_length = parse_max_length(os.environ.get("DECISION_MAX_LENGTH", "256"))
    batch_size = parse_dim(
        "DECISION_ENCODE_BATCH",
        os.environ.get("DECISION_ENCODE_BATCH"),
        default=16,
        lo=1,
        hi=4096,
    )
    device = resolve_device(os.environ.get("DECISION_DEVICE"))
    dtype = resolve_dtype(os.environ.get("DECISION_DTYPE"))
    check_dtype_device(dtype, device)
    encoder = create_state_encoder(
        model_id,
        feature_dim,
        dtype=dtype,
        device=device,
        max_length=max_length,
        batch_size=batch_size,
    )
    encoder.load()
    return DecisionService(
        encoder=encoder,
        dim=feature_dim,
        checkpoint_dir=checkpoint_dir,
        vision_encoder=vision_encoder,
    )


def create_backend(service: DecisionService | None = None) -> Backend:
    """Выбирает бэкенд по окружению: локальные головы или внешний SystemOne-сервис."""
    kind = os.environ.get("DECISION_BACKEND", "local").strip().lower()
    if kind == "http":
        url = os.environ.get("DECISION_BACKEND_URL")
        if not url:
            raise RuntimeError("DECISION_BACKEND=http requires DECISION_BACKEND_URL")
        model = os.environ.get("DECISION_BACKEND_MODEL", "remote")
        modalities = parse_modalities(
            os.environ.get("DECISION_BACKEND_MODALITIES", "text")
        )
        return HttpSystemOneBackend(url, model=model, modalities=modalities)
    if kind == "cascade":
        return create_cascade_backend(service)
    return LocalHeadsBackend(service or create_service())


def create_cascade_backend(service: DecisionService | None = None) -> CascadeBackend:
    """Локальные головы как primary, внешний SystemOne как fallback."""
    url = os.environ.get("DECISION_CASCADE_FALLBACK_URL")
    if not url:
        raise RuntimeError(
            "DECISION_BACKEND=cascade requires DECISION_CASCADE_FALLBACK_URL"
        )
    threshold = float(os.environ.get("DECISION_CASCADE_THRESHOLD", "0.6"))
    primary = LocalHeadsBackend(service or create_service())
    fallback = HttpSystemOneBackend(
        url,
        model=os.environ.get("DECISION_CASCADE_FALLBACK_MODEL", "remote"),
        modalities=parse_modalities(
            os.environ.get("DECISION_CASCADE_FALLBACK_MODALITIES", "text")
        ),
    )
    return CascadeBackend(primary, fallback, threshold=threshold)


app = FastAPI(title="decision-cascade API", version=__version__)

# Единый исполнитель решений для всех эндпоинтов: локальные головы, внешний
# SystemOne-сервис или каскад (по DECISION_BACKEND). Локальный сервис создаётся
# лениво — только если выбранный бэкенд его действительно использует.
backend: Backend = create_backend()


class NoulQuestion(BaseModel):
    type: Literal["noul"]
    instructions: str
    criteria: list[str] | None = Field(None, min_length=2, max_length=2)


class ChoiceQuestion(BaseModel):
    type: Literal["choice"]
    instructions: str
    options: list[str] = Field(..., min_length=2, max_length=255)
    criteria: dict[str, str] | None = Field(None, max_length=255)

    @field_validator("options")
    @classmethod
    def _unique_options(cls, options: list[str]) -> list[str]:
        if len(set(options)) != len(options):
            raise ValueError("options must be unique")
        return options


class ScoreQuestion(BaseModel):
    type: Literal["score"]
    instructions: str
    levels: list[str] = Field(..., min_length=2, max_length=255)
    criteria: list[str] | None = Field(None, max_length=255)

    @field_validator("levels")
    @classmethod
    def _unique_levels(cls, levels: list[str]) -> list[str]:
        if len(set(levels)) != len(levels):
            raise ValueError("levels must be unique")
        return levels

    @model_validator(mode="after")
    def _criteria_match_levels(self) -> ScoreQuestion:
        if self.criteria is not None and len(self.criteria) != len(self.levels):
            raise ValueError("criteria length must match levels length")
        return self


Question = NoulQuestion | ChoiceQuestion | ScoreQuestion


class DecisionRequest(BaseModel):
    state: str = Field(..., min_length=1)
    questions: dict[str, Question] = Field(..., min_length=1)


class DecisionResponse(BaseModel):
    schema_version: str
    decisions: dict[str, Annotated[DecisionResult, Field(discriminator="type")]]


class BatchDecisionRequest(BaseModel):
    states: list[str] = Field(..., min_length=1, max_length=256)
    questions: dict[str, Question] = Field(..., min_length=1)

    @field_validator("states")
    @classmethod
    def _non_empty_states(cls, states: list[str]) -> list[str]:
        if any(not state.strip() for state in states):
            raise ValueError("states must be non-empty strings")
        return states


class BatchDecisionResponse(BaseModel):
    schema_version: str
    results: list[dict[str, Annotated[DecisionResult, Field(discriminator="type")]]]


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


# Все эндпоинты — синхронные (``def``): FastAPI выполняет их в threadpool, поэтому
# тяжёлый CPU-инференс не блокирует event loop и остальные соединения.
@app.post("/decisions", response_model=DecisionResponse)
def create_decisions(req: DecisionRequest) -> DecisionResponse:
    """Оценивает все вопросы параллельно одним прогоном энкодера + голов."""
    result = backend.decide(req.state, _to_specs(req.questions))
    return DecisionResponse(schema_version=SCHEMA_VERSION, decisions=result.decisions)


@app.post("/decisions/batch", response_model=BatchDecisionResponse)
def create_decisions_batch(req: BatchDecisionRequest) -> BatchDecisionResponse:
    """Батч: все состояния и вопросы за один вызов энкодера (внутри — батчами)."""
    results = [
        result.decisions
        for result in backend.decide_many(req.states, _to_specs(req.questions))
    ]
    return BatchDecisionResponse(schema_version=SCHEMA_VERSION, results=results)


@app.post("/v1/systemone", response_model=SystemOneResponse)
def create_systemone_decisions(
    req: SystemOneRequest,
) -> SystemOneResponse | JSONResponse:
    """Стандартный протокол Jev/SystemOne: state + типизированные вопросы -> ответы.

    Неподдерживаемая бэкендом модальность (например, изображения на текстовом
    локальном бэкенде) отклоняется типизированной ошибкой до исполнения.
    """
    requested = requested_modalities(images=req.images, videos=req.videos)
    missing = missing_modalities(requested, backend.capabilities.modalities)
    if missing:
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": "MODALITY_NOT_SUPPORTED",
                    "message": (
                        f"backend {backend.name!r} does not support "
                        f"{sorted(missing)}"
                    ),
                    "supported": sorted(backend.capabilities.modalities),
                    "requested": sorted(requested),
                }
            },
        )
    try:
        specs = systemone_to_specs(req.questions)
    except ValueError as exc:
        return JSONResponse(
            status_code=422,
            content={"error": {"code": "INVALID_QUESTION", "message": str(exc)}},
        )
    result = backend.decide(
        req.state,
        specs,
        images=req.images,
        videos=req.videos,
        media_kwargs=req.media_kwargs,
    )
    answers = {
        name: result_to_answer_model(decision)
        for name, decision in result.decisions.items()
    }
    return SystemOneResponse(
        model=backend.name, answers=answers, usage=result.usage
    )