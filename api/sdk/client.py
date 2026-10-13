"""Python SDK для System One Decision API: типизированный клиент с валидацией."""
from __future__ import annotations

from typing import Annotated, Any, Literal, Self

import httpx
from pydantic import BaseModel, Field

from src.schemas import SCHEMA_VERSION

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


def validate_response(payload: dict[str, Any]) -> list[str] | None:
    """Проверяет ответ API на соответствие типизированному контракту."""
    schema_version = payload.get("schema_version")
    if not isinstance(schema_version, str) or schema_version != SCHEMA_VERSION:
        raise ValueError(
            f"schema_version mismatch: expected {SCHEMA_VERSION}, got {schema_version!r}"
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
        timeout: float = 10.0,
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

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()