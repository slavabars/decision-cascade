"""Risk-aware каскад: эскалация по оценке ``P(ошибка)``, а не по порогу уверенности.

Порог на «сырой» уверенности не равен вероятности ошибки: ``confidence = 0.6`` не
означает «60% верно». Здесь мы обучаем маленький оценщик ``P(ошибка)`` по признакам
решения (уверенность, энтропия, отрыв лучшего класса) и эскалируем во внешнюю
модель только те решения, у которых оценка превышает **бюджет ошибок**.

Роль внешней модели играет заглушка ``StubFallback``: она считает вызовы.
Оценщик риска обучается на синтетическом отложенном наборе прямо в примере —
чтобы пример был полностью офлайн и без скачивания весов.

В конце пример показывает few-shot: как размеченные примеры подключают риск-модель
к конкретному вопросу (в сервисе это ``POST /v1/feedback``).

Запуск:

    .venv/bin/python examples/cascade/risk_aware.py --budget 0.3
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.backends.base import BackendResult, Capabilities
from src.backends.cascade import CascadeBackend, RiskAwareCascadeBackend
from src.risk import ErrorRiskModel
from src.sampler.questions import NoulSpec
from src.schemas import NoulResult
from training.recipe.risk import fit_error_risk_model


def _training_set() -> tuple[list[NoulResult], list[bool]]:
    """Синтетический отложенный набор: уверенные решения чаще верны."""
    probabilities = [
        0.99, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55,
        0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15, 0.10,
    ]
    decisions, correct = [], []
    for p in probabilities:
        # верно, если решение достаточно далеко от границы 0.5
        is_correct = p >= 0.75 or p <= 0.25
        decisions.append(NoulResult(noul=p, confidence=abs(2 * p - 1)))
        correct.append(is_correct)
    return decisions, correct


class StubPrimary:
    """Локальный бэкенд с заранее заданными решениями."""

    name = "local-stub"
    capabilities = Capabilities(modalities=frozenset({"text"}))

    def __init__(self, table: dict[str, NoulResult]) -> None:
        self.table = table

    def decide(
        self, state: str, questions, *, images=None, videos=None, media_kwargs=None
    ) -> BackendResult:
        del state, images, videos, media_kwargs
        return BackendResult(
            decisions={name: self.table[name] for name in questions},
            usage={"calls": 1},
        )


class StubFallback:
    """Заглушка внешней модели: считает, сколько раз её вызвали."""

    name = "stub-llm"
    capabilities = Capabilities(modalities=frozenset({"text"}))

    def __init__(self) -> None:
        self.calls = 0

    def decide(
        self, state: str, questions, *, images=None, videos=None, media_kwargs=None
    ) -> BackendResult:
        del state, images, videos, media_kwargs
        self.calls += len(questions)
        decisions = {
            name: NoulResult(noul=0.99, confidence=0.98) for name in questions
        }
        return BackendResult(decisions=decisions, usage={"calls": 1})


def main() -> None:
    parser = argparse.ArgumentParser(description="Risk-aware cascade demo")
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--budget", type=float, default=0.3)
    args = parser.parse_args()

    decisions, correct = _training_set()
    model = fit_error_risk_model(decisions, correct, seed=0)
    print(f"оценщик P(ошибка) обучен на {len(decisions)} решениях\n")

    table = {
        "q1-уверенный": NoulResult(noul=0.95, confidence=abs(2 * 0.95 - 1)),
        "q2-уверенный": NoulResult(noul=0.80, confidence=abs(2 * 0.80 - 1)),
        "q3-спорный": NoulResult(noul=0.60, confidence=abs(2 * 0.60 - 1)),
        "q4-спорный": NoulResult(noul=0.52, confidence=abs(2 * 0.52 - 1)),
    }
    questions = {name: NoulSpec(f"?{name}") for name in table}
    primary = StubPrimary(table)

    threshold_backend = CascadeBackend(
        primary, StubFallback(), threshold=args.threshold
    )
    threshold_result = threshold_backend.decide("state", questions)

    risk_fallback = StubFallback()
    risk_backend = RiskAwareCascadeBackend(
        primary, risk_fallback, model, risk_budget=args.budget
    )
    risk_result = risk_backend.decide("state", questions)

    print(f"{'вопрос':<14} {'noul':>6} {'conf':>6} {'P(ошиб)':>8}  решение")
    for name, decision in table.items():
        error = model.predict_error(decision)
        if risk_backend._uncertain(decision):
            action = "эскалация (risk)"
        else:
            action = "локально"
        print(
            f"{name:<14} {decision.noul:>6.2f} {decision.confidence:>6.2f} "
            f"{error:>8.3f}  {action}"
        )

    print()
    print(
        f"порог {args.threshold}: эскалировано "
        f"{threshold_result.usage['escalated']} из {len(questions)} "
        f"(confidence < {args.threshold})"
    )
    print(
        f"бюджет {args.budget}: эскалировано "
        f"{risk_result.usage['escalated']} из {len(questions)} "
        f"(P(ошибка) > {args.budget}, reason={risk_result.usage.get('reason')})"
    )

    _few_shot_demo(primary, model)


def _few_shot_demo(primary: StubPrimary, model: ErrorRiskModel) -> None:
    """Разметка новых примеров -> риск-модель вопроса переобучается.

    Тот же шов, что в сервисе: примеры приходят через ``POST /v1/feedback``,
    здесь — программно через ``add_examples``. Набрав порог примеров, каскад
    переобучает риск-модель по схеме {чужие строки + свои} и применяет её только
    к своему вопросу. Hard gate при этом не ослабляется (см. docs).
    """
    decisions, correct = _training_set()
    few_shot = RiskAwareCascadeBackend(
        primary, StubFallback(), model, risk_budget=0.3, few_shot_n=2
    )
    print(f"\nразмеченные примеры (порог few_shot_n={few_shot.few_shot_n}):")
    for question in ("q4-спорный", "q3-спорный"):
        for decision, is_correct in zip(decisions[:2], correct[:2]):
            model_q = few_shot.add_examples(question, [decision], [is_correct])
            counts = few_shot.example_counts
            rows, _labels = few_shot.reference_rows(question)
            reference = 0 if rows is None else len(rows)
            print(
                f"  {question}: примеров {counts[question]}, чужих строк "
                f"{reference}, модель {'переобучена' if model_q else 'базовая'}"
            )
    print(
        "  первая пачка шла без чужих строк — обучение только на своих: "
        "измеримо хуже, пока примеров мало (см. docs/risk-aware-cascade.md)"
    )


if __name__ == "__main__":
    main()
