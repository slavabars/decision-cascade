"""Слабая разметка: пак из ответов сильной модели на неразмеченных данных.

Идея: разметка руками дорога, а сильная модель уже умеет отвечать на наши
типизированные вопросы. Прогоняем её по неразмеченным состояниям, берём её
решения как метки и обучаем маленькие головы на этих метках. Получается
дистилляция поведения учителя в дешёвый локальный движок.

Что важно понимать:

- это **не** ground truth. Метки наследуют ошибки и смещения учителя; качество
  головы ограничено качеством учителя на вашем домене;
- проверять результат нужно на настоящей разметке (``eval.report.run`` или
  свой test-сплит), а не на метках учителя — иначе измеряем согласие с собой;
- низкая ``confidence`` учителя — сигнал выбросить пример
  (``--min-confidence``), а не «доучить» его.

Пример:
    python -m training.labeling.run --input data/unlabeled.jsonl \
        --out data/weak/support.jsonl --specs support \
        --teacher-url http://127.0.0.1:9000 --teacher-model clef
    python -m training.recipe.train_heads --dataset data/weak/support.jsonl \
        --out-dir checkpoints/weak-support --encoder hash --dim 64

Вход — JSONL со строкой состояния в поле ``state`` (или ``text``); строка, не
похожая на JSON, тоже считается состоянием. Выход — тот же формат, что у
``eval.datasets``: ``{"id", "state", "labels", ...}``, поэтому его напрямую
принимает ``training.recipe.train_heads --dataset``.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

from src.backends.base import BackendResult
from src.sampler.questions import (
    ChoiceSpec,
    NoulSpec,
    QuestionSpec,
    ScoreSpec,
    spec_to_dict,
)
from src.schemas import DecisionResult

Decide = Callable[[str, dict[str, QuestionSpec]], BackendResult]

SPEC_CHOICES = ("support", "public", "routerbench")


def load_specs(name: str) -> dict[str, QuestionSpec]:
    """Набор вопросов по имени — тот же, что у ``train_heads --specs``."""
    if name == "support":
        from eval.datasets.text_dataset import QUESTIONS, spec_from_api

        data = QUESTIONS
    elif name == "public":
        from eval.datasets.public_dataset import PUBLIC_QUESTIONS as data
        from eval.datasets.text_dataset import spec_from_api
    elif name == "routerbench":
        from eval.datasets.routerbench import ROUTERBENCH_QUESTIONS as data
        from eval.datasets.text_dataset import spec_from_api
    else:
        raise ValueError(f"unknown specs {name!r}; expected one of {SPEC_CHOICES}")
    return {key: spec_from_api(value) for key, value in data.items()}


def read_states(path: str | Path) -> list[dict[str, Any]]:
    """Читает состояния из JSONL (``state``/``text``) или построчного текста.

    Пустые строки пропускаются. Если строка не парсится как JSON-объект, она
    целиком считается состоянием — так можно размечать обычный ``.txt``.
    """
    path = Path(path)
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for index, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                payload = None
            if isinstance(payload, dict):
                state = payload.get("state") or payload.get("text")
                if not isinstance(state, str) or not state.strip():
                    raise ValueError(
                        f"{path}:{index + 1}: object has no non-empty 'state'/'text'"
                    )
                rows.append(
                    {
                        "id": str(payload.get("id", f"weak-{index}")),
                        "state": state,
                    }
                )
            elif isinstance(payload, str):
                rows.append({"id": f"weak-{index}", "state": payload})
            else:
                rows.append({"id": f"weak-{index}", "state": line})
    return rows


def _label(spec: QuestionSpec, decision: DecisionResult, name: str) -> Any:
    """Метка в формате обучения: 0/1 для noul, строка для choice, индекс для score."""
    if isinstance(spec, NoulSpec):
        return 1 if decision.noul >= 0.5 else 0
    if isinstance(spec, ChoiceSpec):
        if decision.choice not in spec.options:
            raise ValueError(
                f"teacher returned {decision.choice!r} for {name!r}, "
                f"expected one of {list(spec.options)}"
            )
        return decision.choice
    if isinstance(spec, ScoreSpec):
        probabilities = decision.probabilities or {}
        if probabilities:
            # берём наиболее вероятный уровень: это и есть метка-класс
            return max(
                range(len(spec.levels)),
                key=lambda i: probabilities.get(spec.levels[i], 0.0),
            )
        index = round(decision.score)
        return min(max(index, 0), len(spec.levels) - 1)
    raise TypeError(f"unsupported question type: {type(spec).__name__}")


def label_examples(
    rows: list[dict[str, Any]],
    specs: dict[str, QuestionSpec],
    decide: Decide,
    *,
    teacher: str = "teacher",
    min_confidence: float = 0.0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Размечает строки учителем и возвращает (примеры, статистику).

    Пример выбрасывается, если минимальная уверенность учителя по вопросам
    ниже ``min_confidence``. Итоговая статистика считает и отброшенные строки,
    чтобы было видно, насколько учитель уверен на домене.
    """
    if not 0.0 <= min_confidence <= 1.0:
        raise ValueError(f"min_confidence must be in [0, 1], got {min_confidence}")
    examples: list[dict[str, Any]] = []
    dropped = 0
    confidences: dict[str, list[float]] = {name: [] for name in specs}
    for row in rows:
        result = decide(row["state"], specs)
        missing = set(specs) - set(result.decisions)
        if missing:
            raise ValueError(
                f"teacher did not answer questions {sorted(missing)} for {row['id']}"
            )
        confidences_row = {
            name: float(result.decisions[name].confidence) for name in specs
        }
        if min_confidence and min(confidences_row.values()) < min_confidence:
            dropped += 1
            continue
        for name, value in confidences_row.items():
            confidences[name].append(value)
        examples.append(
            {
                "id": row["id"],
                "state": row["state"],
                "labels": {
                    name: _label(specs[name], result.decisions[name], name)
                    for name in specs
                },
                "teacher": teacher,
                "teacher_confidence": confidences_row,
            }
        )
    report = {
        "teacher": teacher,
        "input": len(rows),
        "kept": len(examples),
        "dropped": dropped,
        "min_confidence": min_confidence,
        "mean_confidence": {
            name: (sum(values) / len(values) if values else 0.0)
            for name, values in confidences.items()
        },
        "label_distribution": {
            name: dict(Counter(str(example["labels"][name]) for example in examples))
            for name in specs
        },
    }
    return examples, report


def _build_teacher(args: argparse.Namespace):
    """Внешний SystemOne-бэкенд (сильная модель) по аргументам CLI."""
    from src.backends.http import HttpSystemOneBackend

    if not args.teacher_url:
        raise ValueError("--teacher-url is required (SystemOne endpoint of teacher)")
    return HttpSystemOneBackend(
        args.teacher_url,
        model=args.teacher_model,
        timeout=args.teacher_timeout,
        retries=args.teacher_retries,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Weak labeling via SystemOne teacher")
    parser.add_argument("--input", type=Path, required=True, help="JSONL/TXT состояния")
    parser.add_argument("--out", type=Path, required=True, help="JSONL с метками")
    parser.add_argument("--specs", choices=SPEC_CHOICES, default="support")
    parser.add_argument("--limit", type=int, default=0, help="взять первые N строк")
    parser.add_argument("--teacher-url", default=None)
    parser.add_argument("--teacher-model", default="teacher")
    parser.add_argument("--teacher-timeout", type=float, default=60.0)
    parser.add_argument("--teacher-retries", type=int, default=1)
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.0,
        help="выбросить пример, если уверенность учителя ниже порога",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="куда записать статистику (по умолчанию рядом с --out, .report.json)",
    )
    return parser


def main(
    argv: list[str] | None = None,
    *,
    decide: Decide | None = None,
    teacher: str | None = None,
) -> dict[str, Any]:
    """CLI слабой разметки; ``decide`` подменяет учителя в тестах."""
    args = _parser().parse_args(argv)
    specs = load_specs(args.specs)
    rows = read_states(args.input)
    if args.limit > 0:
        rows = rows[: args.limit]
    if not rows:
        raise ValueError(f"no states in {args.input}")

    teacher_name = teacher or args.teacher_model
    decide_fn = decide if decide is not None else _build_teacher(args).decide
    examples, report = label_examples(
        rows,
        specs,
        decide_fn,
        teacher=teacher_name,
        min_confidence=args.min_confidence,
    )
    report["specs"] = args.specs
    report["questions"] = {name: spec_to_dict(spec) for name, spec in specs.items()}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for example in examples:
            fh.write(json.dumps(example, ensure_ascii=False) + "\n")
    report_path = args.report or args.out.with_suffix(".report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"labeled {report['kept']}/{report['input']} "
        f"(dropped {report['dropped']}) -> {args.out}",
        flush=True,
    )
    print(f"report -> {report_path}", flush=True)
    return report


if __name__ == "__main__":
    main()
    sys.exit(0)
