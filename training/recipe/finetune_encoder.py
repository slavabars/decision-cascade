"""Рецепт дообучения энкодера состояния под задачу решений.

Дообучает HF-энкодер (и головы решений) на train-сплите и сохраняет энкодер
как обычный каталог ``from_pretrained``: отдельного формата нет, поэтому
дообученный энкодер подставляется и как ``--encoder`` в ``eval.risk.features``,
и как ``DECISION_MODEL_ID`` сервиса. Разбиение на train/val берётся тем же
помощником, что и в риск-харнессе (``eval.report.run._load_splits``), иначе
замер шёл бы на строках, которые энкодер видел при обучении.

Веса никуда не публикуются: каталог остаётся в ``checkpoints/`` (в git не идёт).
Метаданные прогона (шаги, lr, сплиты, потери, точность голов) пишутся рядом,
чтобы замер можно было воспроизвести.

Пример:
    python -m training.recipe.finetune_encoder \
        --specs routerbench --encoder openai-community/gpt2 --dim 768 \
        --limit 4000 --steps 300 --lr 1e-4 --risk-steps 200 \
        --out-dir checkpoints/finetune/routerbench
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

from eval.benchmark.core import evaluate_benchmark
from eval.report.run import _encoder, _load_splits, _specs
from src.decision_head.heads import DecisionHeads
from src.encoder.base import TrainableStateEncoder
from src.sampler.questions import questions_signature, spec_to_dict
from src.schemas import SCHEMA_VERSION
from training.finetune.train import FinetuneTrainer


def _encode_batch() -> int:
    """Размер чанка энкодера: батч тренера должен совпадать с ним (см. тренер)."""
    return int(os.environ.get("DECISION_ENCODE_BATCH", "16"))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Finetune state encoder (recipe)")
    parser.add_argument(
        "--specs",
        choices=["support", "public", "routerbench"],
        default="routerbench",
        help="набор вопросов (как в eval.risk.features)",
    )
    parser.add_argument("--dataset", type=Path, default=None, help="каталог/файл JSONL")
    parser.add_argument(
        "--limit", type=int, default=0, help="ограничение train-сплита (0 — целиком)"
    )
    parser.add_argument(
        "--encoder",
        default="openai-community/gpt2",
        help="model_id HF (хэш-энкодер дообучать нечего)",
    )
    parser.add_argument("--dim", type=int, default=768)
    parser.add_argument(
        "--steps", type=int, default=300, help="шаги фазы задачи (энкодер + головы)"
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--risk-steps",
        type=int,
        default=0,
        help="шаги фазы ошибки: энкодер учится предсказывать ошибку (0 — выключено)",
    )
    parser.add_argument(
        "--risk-lr", type=float, default=None, help="lr фазы ошибки (по умолчанию --lr)"
    )
    parser.add_argument("--batch-size", type=int, default=_encode_batch())
    parser.add_argument("--device", default=None, help="auto|cpu|cuda|cuda:N|mps")
    parser.add_argument(
        "--dtype",
        default=None,
        help="float32 по умолчанию; bfloat16/float16 экономят память на GPU",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser


def _loss_summary(losses: list[float]) -> dict[str, float] | None:
    if not losses:
        return None
    return {"first": losses[0], "last": losses[-1], "min": min(losses)}


def _report(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "overall_accuracy": report["overall_accuracy"],
        "questions": report["questions"],
    }


def main(argv: list[str] | None = None) -> dict:
    args = _parser().parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    specs = _specs(args.specs)
    splits = _load_splits(
        SimpleNamespace(
            specs=args.specs,
            dataset=args.dataset,
            limit=args.limit,
            seed=args.seed,
            shot="0shot",
        )
    )
    train = splits["train"]
    val = splits.get("val") or []

    encoder = _encoder(args.encoder, args.dim, args.device, args.dtype)
    if not isinstance(encoder, TrainableStateEncoder):
        raise SystemExit(
            f"энкодер {args.encoder!r} не поддерживает дообучение: нужен "
            "TrainableStateEncoder (хэш-энкодер учить нечему)"
        )
    heads = DecisionHeads(args.dim, specs, seed=args.seed)
    trainer = FinetuneTrainer(encoder, heads, lr=args.lr, seed=args.seed)
    task_losses = trainer.fit(
        specs, train, steps=args.steps, batch_size=args.batch_size
    )
    risk_losses: list[float] = []
    if args.risk_steps:
        risk_losses = trainer.fit_error(
            specs,
            train,
            steps=args.risk_steps,
            lr=args.risk_lr,
            batch_size=args.batch_size,
        )
    # считать параметры надо до set_trainable(False): после него список пуст
    n_trainable = sum(param.numel() for param in encoder.trainable_parameters())
    # eval-режим возвращаем до сохранения: без него в чекпоинт уехали бы
    # признаки с dropout, не совпадающие с инференсом
    encoder.set_trainable(False)
    # головы обучались на устройстве энкодера, а в рантайме признаки приходят
    # с CPU (``encode`` отдаёт их на CPU) — возвращаем головы туда же
    heads.cpu()
    encoder_dir = encoder.save_pretrained(out_dir / "encoder")
    heads_path = heads.save_checkpoint(out_dir / "heads.pt")
    if trainer.error_head is not None:
        torch.save(
            trainer.error_head.state_dict(), out_dir / "error_head.pt"
        )

    metrics = {"train": _report(evaluate_benchmark(specs, train, encoder, heads))}
    if val:
        metrics["val"] = _report(evaluate_benchmark(specs, val, encoder, heads))

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "encoder": args.encoder,
        "encoder_dir": str(encoder_dir),
        "feature_dim": args.dim,
        "specs": args.specs,
        "questions_signature": questions_signature(specs),
        "seed": args.seed,
        "dataset": str(args.dataset) if args.dataset is not None else None,
        "splits": {name: len(part) for name, part in splits.items()},
        "steps": args.steps,
        "lr": args.lr,
        "risk_steps": args.risk_steps,
        "risk_lr": args.risk_lr,
        "batch_size": args.batch_size,
        "trainable_parameters": n_trainable,
        "losses": {
            "task": _loss_summary(task_losses),
            "risk": _loss_summary(risk_losses),
        },
        "metrics": metrics,
        "questions": {name: spec_to_dict(spec) for name, spec in specs.items()},
    }
    meta_path = out_dir / "metadata.json"
    meta_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(f"encoder  -> {encoder_dir}")
    print(f"heads    -> {heads_path}")
    print(f"metadata -> {meta_path}")
    for name, report in metrics.items():
        print(
            f"  {name:<5} acc={report['overall_accuracy']:.3f} "
            f"(n={len(splits[name])})"
        )
    print(
        f"замер: python -m eval.risk.features --specs {args.specs} "
        f"--encoder {encoder_dir} --dim {args.dim} --state --loo"
    )
    return metadata


if __name__ == "__main__":
    main()
