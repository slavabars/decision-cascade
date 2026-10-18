"""Python SDK для decision-cascade API: типизированный клиент с валидацией."""
from __future__ import annotations

from typing import Annotated, Any, Literal, Self

import httpx
from pydantic import BaseModel, Field

from src.schemas import SCHEMA_VERSION
from src.versioning import is_compatible

QuestionPayload = dict[str, Any]


class _NoulOut(BaseModel):
    type: Literal["noul"] = "noul"
    noul: float = Field(..., ge=0.0, le=1.0)
    confidence: float = Field(..., ge=0.0, le=1.0)


class _ChoiceOut(BaseModel):
    type: Literal["choice"] = "choice"
    choice: str
    probabilities: dict[str, float]
    confidence: float = Field(..., ge=0.0, le=1.0)


class _ScoreOut(BaseModel):
    type: Literal["score"] = "score"
    score: float
    probabilities: dict[str, float]
    confidence: float = Field(..., ge=0.0, le=1.0)
    legend: list[str]


_Out = Annotated[
    _NoulOut | _ChoiceOut | _ScoreOut, Field(discriminator="type")
]


class _Response(BaseModel):
    schema_version: str
    decisions: dict[str, _Out]


class _FeedbackQuestion(BaseModel):
    question: str
    accepted: int = Field(..., ge=0)
    examples: int = Field(..., ge=0)
    reference: int = Field(..., ge=0)
    retrained: bool


class _FeedbackResponse(BaseModel):
    accepted: int = Field(..., ge=0)
    questions: list[_FeedbackQuestion]


def validate_response(payload: dict[str, Any]) -> list[str] | None:
    """Проверяет ответ API на соответствие типизированному контракту.

    Версия схемы должна быть совместима по major (аддитивные minor-изменения не
    ломают клиент), см. ``src/versioning.py``.
    """
    schema_version = payload.get("schema_version")
    if not isinstance(schema_version, str) or not is_compatible(
        SCHEMA_VERSION, schema_version
    ):
        raise ValueError(
            f"schema_version mismatch: expected major of {SCHEMA_VERSION}, "
            f"got {schema_version!r}"
        )
    _Response.model_validate(payload)
    for name, res in payload["decisions"].items():
        probs = res.get("probabilities")
        if probs is not None:
            if abs(sum(probs.values()) - 1.0) > 1e-6:
                raise ValueError(f"decision {name!r}: probabilities must sum to 1.0")
            if any(p < 0.0 for p in probs.values()):
                raise ValueError(f"decision {name!r}: probabilities must be non-negative")
    return None


class DecisionClient:
    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        timeout: float = 60.0,
        transport: httpx.BaseTransport | None = None,
    ):
        self._client = httpx.Client(base_url=base_url, timeout=timeout, transport=transport)

    def decisions(
        self, state: str, questions: dict[str, QuestionPayload]
    ) -> dict[str, Any]:
        payload = {"state": state, "questions": questions}
        resp = self._client.post("/decisions", json=payload)
        resp.raise_for_status()
        body = resp.json()
        validate_response(body)
        return body

    def feedback(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        """Прислать размеченные примеры решений (few-shot риск-модели).

        ``examples`` — список вида
        ``{"question": ..., "decision": <решение из decisions()>, "correct": bool}``:
        повторяется именно решение, которое сервер уже отдавал, состояние кодировать
        заново не нужно. В ответе — сколько примеров накоплено по вопросу и не
        достигнут ли порог `DECISION_RISK_FEW_SHOT_N` (тогда риск-модель вопроса
        переобучена). Если бэкенд обратную связь не принимает, сервер отвечает
        ``409 FEEDBACK_UNSUPPORTED`` — это ``httpx.HTTPStatusError``.
        """
        resp = self._client.post("/v1/feedback", json={"examples": examples})
        resp.raise_for_status()
        body = resp.json()
        _FeedbackResponse.model_validate(body)
        return body

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()