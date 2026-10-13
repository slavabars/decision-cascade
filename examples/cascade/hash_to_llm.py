"""Двухуровневый каскад in-process: дешёвый локальный бэкенд -> «LLM».

Показывает ``CascadeBackend``: всё, что локальные головы решают уверенно,
остаётся локально; низкоуверенные вопросы уходят в fallback. Роль внешней
модели здесь играет заглушка ``StubFallback`` — она печатает, когда её вызвали,
и возвращает «уверенные» решения. В продакшене fallback — это
``HttpSystemOneBackend`` (см. ``examples/protocol/gateway.sh``).

Запуск:

    .venv/bin/python examples/cascade/hash_to_llm.py --threshold 0.6
"""
import argparse
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples._common import build_service, primary_value, ticket_specs
from src.backends.base import BackendResult, Capabilities
from src.backends.cascade import CascadeBackend
from src.backends.local import LocalHeadsBackend
from src.sampler.questions import ChoiceSpec, NoulSpec, QuestionSpec, ScoreSpec
from src.schemas import ChoiceResult, NoulResult, ScoreResult

STATE = "Не приходит выплата Stripe третьи сутки, теряем выручку!"


class StubFallback:
    """Заглушка внешней модели: уверенные решения + счётчик вызовов."""

    name = "stub-llm"
    capabilities = Capabilities(modalities=frozenset({"text"}))

    def __init__(self) -> None:
        self.calls = 0

    def decide(
        self,
        state: str,
        questions: dict[str, QuestionSpec],
        *,
        images: list[Any] | None = None,
        videos: list[Any] | None = None,
        media_kwargs: dict[str, Any] | None = None,
    ) -> BackendResult:
        self.calls += 1
        decisions = {
            name: _confident(spec) for name, spec in questions.items()
        }
        return BackendResult(
            decisions=decisions, usage={"model": self.name, "calls": 1}
        )


def _confident(spec: QuestionSpec):
    """Детерминированное «уверенное» решение для любой спецификации."""
    if isinstance(spec, NoulSpec):
        return NoulResult(noul=0.9, confidence=0.8)
    if isinstance(spec, ChoiceSpec):
        share = 1.0 / len(spec.options)
        return ChoiceResult(
            choice=spec.options[0],
            probabilities={option: share for option in spec.options},
            confidence=1.0,
        )
    if isinstance(spec, ScoreSpec):
        share = 1.0 / len(spec.levels)
        return ScoreResult(
            score=float(len(spec.levels) - 1),
            probabilities={level: share for level in spec.levels},
            confidence=1.0,
            legend=spec.levels,
        )
    raise TypeError(f"unsupported spec: {type(spec).__name__}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Two-tier cascade in-process")
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    primary = LocalHeadsBackend(build_service(seed=args.seed), name="local-hash")
    fallback = StubFallback()
    cascade = CascadeBackend(primary, fallback, threshold=args.threshold)

    result = cascade.decide(STATE, ticket_specs())
    print(f"каскад: {cascade.name}, порог={args.threshold}")
    print(f"вызовов fallback: {fallback.calls}, эскалировано: {result.usage['escalated']}")
    print(f"команда: {primary_value(result.decisions, 'команда')}")


if __name__ == "__main__":
    main()
