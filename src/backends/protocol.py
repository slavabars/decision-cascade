"""Чистые конвертеры протокола Jev/SystemOne (без pydantic).

Внутренние типизированные решения <-> словари ответов протокола, а также
внутренние спецификации вопросов -> тело запроса к внешнему SystemOne-сервису.
"""
from __future__ import annotations

from typing import Any

from src.sampler.questions import ChoiceSpec, NoulSpec, QuestionSpec, ScoreSpec
from src.schemas import ChoiceResult, DecisionResult, NoulResult, ScoreResult


def result_to_answer(result: DecisionResult) -> dict[str, Any]:
    """Внутреннее решение -> ответ протокола (noul использует ключ ``probability``)."""
    if isinstance(result, NoulResult):
        return {
            "type": "noul",
            "probability": result.noul,
            "confidence": result.confidence,
        }
    if isinstance(result, ChoiceResult):
        return {
            "type": "choice",
            "choice": result.choice,
            "confidence": result.confidence,
            "probabilities": dict(result.probabilities or {}),
        }
    if isinstance(result, ScoreResult):
        return {
            "type": "score",
            "score": result.score,
            "confidence": result.confidence,
            "probabilities": dict(result.probabilities or {}),
            "legend": list(result.legend),
        }
    raise TypeError(f"unsupported decision result: {type(result).__name__}")


def answer_to_result(answer: dict[str, Any]) -> DecisionResult:
    """Ответ протокола -> внутреннее типизированное решение."""
    kind = answer.get("type")
    if kind == "noul":
        probability = answer.get("probability", answer.get("noul"))
        if probability is None:
            raise ValueError("noul answer requires 'probability'")
        confidence = answer.get("confidence")
        if confidence is None:
            confidence = abs(2.0 * float(probability) - 1.0)
        return NoulResult(noul=float(probability), confidence=float(confidence))
    if kind == "choice":
        return ChoiceResult(
            choice=answer["choice"],
            probabilities=dict(answer["probabilities"]),
            confidence=float(answer["confidence"]),
        )
    if kind == "score":
        return ScoreResult(
            score=float(answer["score"]),
            probabilities=dict(answer["probabilities"]),
            confidence=float(answer["confidence"]),
            legend=tuple(answer["legend"]),
        )
    raise ValueError(f"unsupported answer type: {kind!r}")


def spec_to_request(spec: QuestionSpec) -> dict[str, Any]:
    """Внутренняя спецификация вопроса -> тело вопроса для SystemOne-сервиса."""
    request: dict[str, Any] = {"type": _spec_type(spec), "instructions": spec.instructions}
    if isinstance(spec, NoulSpec):
        if spec.criteria:
            request["criteria"] = list(spec.criteria)
    elif isinstance(spec, ChoiceSpec):
        request["options"] = list(spec.options)
        if spec.criteria:
            request["criteria"] = dict(spec.criteria)
    elif isinstance(spec, ScoreSpec):
        request["levels"] = list(spec.levels)
        if spec.criteria:
            request["criteria"] = list(spec.criteria)
    return request


def _spec_type(spec: QuestionSpec) -> str:
    if isinstance(spec, NoulSpec):
        return "noul"
    if isinstance(spec, ChoiceSpec):
        return "choice"
    if isinstance(spec, ScoreSpec):
        return "score"
    raise TypeError(f"unsupported question spec: {type(spec).__name__}")
