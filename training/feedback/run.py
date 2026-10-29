"""Дообучение голов на обратной связи: replay + гейт «не навреди».

Дрейф домена — не гипотеза: головы, обученные на срезе прошлого квартала,
теряют точность на новых данных, а калибровка уезжает. Здесь свежие размеченные
примеры (формат ``eval.datasets``: ``state`` + ``labels``) смешиваются с частью
старого датасета (replay) и дообучают головы. Новый пак принимается только если
на **отложенной** части свежих примеров стало не хуже порога, а на старом
домене точность не упала больше допуска — иначе старые головы остаются как есть.

Пример:
    python -m training.feedback.run --pack checkpoints/packs/public-gpt2 \
        --feedback data/feedback.jsonl --out checkpoints/packs/public-gpt2-v2 \
        --replay 0.5 --steps 100

Что важно:

- ``--feedback`` — не «сырые» клики, а решения с метками: клиент повторяет
  состояние и правильные ответы (тот же JSONL, что у ``train_heads``);
- replay обязателен на практике: без него головы забывают старый домен
  (catastrophic forgetting), поэтому по умолчанию берётся половина исходного
  ``dataset.jsonl`` из пака;
- температура из пака переносится как есть, но после дообучения она уже не
  откалибрована — перекалибруйте (``training.recipe.calibrate_temperature``).
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path
from random import Random
from typing import Any

import torch

from eval.benchmark.core import build_targets, encode_examples, evaluate_benchmark
from src.decision_head.heads import DecisionHeads
from src.encoder.caching import CachedEncoder
from src.sampler.questions import ChoiceSpec, NoulSpec, QuestionSpec, ScoreSpec
from training.sft.train import SFTTrainer

#: Допуск по умолчанию: старая точность может просесть не больше чем на 1 п.п.
DEFAULT_MAX_OLD_DROP = 0.01
#: Минимальный прирост на новом домене, чтобы принять пак.
DEFAULT_MIN_NEW_GAIN = 0.0


def validate_label(spec: QuestionSpec, value: Any, name: str) -> Any:
    """Проверяет метку по спецификации вопроса и возвращает её в нужном виде."""
    if isinstance(spec, NoulSpec):
        if value not in (0, 1, 0.0, 1.0, True, False):
            raise ValueError(f"{name}: noul label must be 0/1, got {value!r}")
        return int(bool(value))
    if isinstance(spec, ChoiceSpec):
        if value not in spec.options:
            raise ValueError(
                f"{name}: choice label {value!r} is not one of {list(spec.options)}"
            )
        return value
    if isinstance(spec, ScoreSpec):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name}: score label must be an int, got {value!r}")
        if not 0 <= value < len(spec.levels):
            raise ValueError(
                f"{name}: score label {value} is out of range 0..{len(spec.levels) - 1}"
            )
        return value
    raise TypeError(f"{name}: unsupported question type {type(spec).__name__}")


def read_examples(
    path: str | Path, specs: dict[str, QuestionSpec]
) -> list[dict[str, Any]]:
    """Читает JSONL ``{"state", "labels"}`` и проверяет метки по спецификациям."""
    path = Path(path)
    examples: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for index, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            state = payload.get("state")
            if not isinstance(state, str) or not state.strip():
                raise ValueError(f"{path}:{index + 1}: no non-empty 'state'")
            labels = payload.get("labels")
            if not isinstance(labels, dict):
                raise TypeError(f"{path}:{index + 1}: no 'labels' object")
            missing = set(specs) - set(labels)
            if missing:
                raise ValueError(
                    f"{path}:{index + 1}: no labels for questions {sorted(missing)}"
                )
            unknown = set(labels) - set(specs)
            if unknown:
                raise ValueError(
                    f"{path}:{index + 1}: labels for unknown questions {sorted(unknown)}"
                )
            examples.append(
                {
                    "id": str(payload.get("id", f"{path.stem}-{index}")),
                    "state": state,
                    "labels": {
                        name: validate_label(spec, labels[name], name)
                        for name, spec in specs.items()
                    },
                }
            )
    if not examples:
        raise ValueError(f"no examples in {path}")
    return examples


def split_holdout(
    examples: list[dict[str, Any]], fraction: float, seed: int = 0
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Делит обратную связь на обучающую и отложенную (для гейта)."""
    if not 0.0 < fraction < 1.0:
        raise ValueError(f"holdout fraction must be in (0, 1), got {fraction}")
    if len(examples) < 5:
        raise ValueError(
            f"feedback is too small for a gate: {len(examples)} examples, need >= 5"
        )
    shuffled = list(examples)
    Random(seed).shuffle(shuffled)
    cut = max(1, round(len(shuffled) * fraction))
    cut = min(cut, len(shuffled) - 1)
    holdout, train = shuffled[:cut], shuffled[cut:]
    return train, holdout


def split_retention(
    examples: list[dict[str, Any]], fraction: float, seed: int = 0
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Делит старый датасет на обучающую часть (replay) и отложенную (retention)."""
    if len(examples) < 10:
        return list(examples), []
    shuffled = list(examples)
    Random(seed).shuffle(shuffled)
    cut = max(1, round(len(shuffled) * min(fraction, 0.5)))
    return shuffled[cut:], shuffled[:cut]


def sample_replay(
    examples: list[dict[str, Any]], fraction: float, seed: int = 0
) -> list[dict[str, Any]]:
    """Берёт долю старого датасета (или повторяет его, если доля больше 1).

    ``fraction > 1`` нужен, когда свежих примеров много: при фиксированном
    replay старый домен тонет в новых и проседает. Повтор строк — дешёвый способ
    удержать баланс без новых данных.
    """
    if fraction < 0.0:
        raise ValueError(f"replay fraction must be >= 0, got {fraction}")
    if fraction == 0.0 or not examples:
        return []
    count = max(1, round(len(examples) * fraction))
    shuffled = list(examples)
    Random(seed + 1).shuffle(shuffled)
    if count <= len(shuffled):
        return shuffled[:count]
    repeats = count // len(shuffled) + 1
    return (shuffled * repeats)[:count]


def retrain(
    heads: DecisionHeads,
    specs: dict[str, QuestionSpec],
    encoder,
    new_examples: list[dict[str, Any]],
    replay_examples: list[dict[str, Any]],
    *,
    steps: int = 100,
    lr: float = 0.05,
    batch_size: int = 32,
    seed: int = 0,
) -> DecisionHeads:
    """Копия голов, дообученная на свежих примерах + replay.

    Копия важна: если гейт не пропустит пак, исходные головы должны остаться
    нетронутыми. Пустые ``new_examples`` (только replay) — контроль: столько же
    шагов на старых данных, без обратной связи.
    """
    if not new_examples and not replay_examples:
        raise ValueError("nothing to train on: no feedback and no replay")
    if new_examples:
        features_new = encode_examples(specs, new_examples, encoder)
        targets_new = {
            name: build_targets(spec, [ex["labels"][name] for ex in new_examples])
            for name, spec in specs.items()
        }
    else:
        features_new, targets_new = {}, {}
    features, targets = features_new, targets_new
    if replay_examples and not new_examples:
        features_old = encode_examples(specs, replay_examples, encoder)
        targets = {
            name: build_targets(spec, [ex["labels"][name] for ex in replay_examples])
            for name, spec in specs.items()
        }
        features = features_old
    elif replay_examples:
        features_old = encode_examples(specs, replay_examples, encoder)
        targets_old = {
            name: build_targets(spec, [ex["labels"][name] for ex in replay_examples])
            for name, spec in specs.items()
        }
        features = {
            name: torch.cat([features_new[name], features_old[name]])
            for name in specs
        }
        targets = {
            name: torch.cat([targets_new[name], targets_old[name]]) for name in specs
        }

    trained = copy.deepcopy(heads)
    torch.manual_seed(seed)
    SFTTrainer(trained, lr=lr).fit(
        features=features,
        targets=targets,
        steps=steps,
        batch_size=batch_size,
    )
    return trained


def summarize(report: dict[str, Any]) -> dict[str, float]:
    """Сжимает отчёт ``evaluate_benchmark`` до accuracy и среднего ECE."""
    per_question = report["questions"]
    return {
        "accuracy": float(report["overall_accuracy"]),
        "ece": float(
            sum(metric["ece"] for metric in per_question.values()) / len(per_question)
        ),
    }


def gate(
    *,
    before_new: dict[str, float],
    after_new: dict[str, float],
    before_old: dict[str, float] | None,
    after_old: dict[str, float] | None,
    min_new_gain: float = DEFAULT_MIN_NEW_GAIN,
    max_old_drop: float = DEFAULT_MAX_OLD_DROP,
) -> dict[str, Any]:
    """Вердикт «не навреди»: новый домен не хуже порога, старый не просел.

    ``before_old``/``after_old`` — ``None``, если старого среза нет (тогда
    retention не проверяется, и это честно написано в отчёте).
    """
    new_gain = float(after_new["accuracy"] - before_new["accuracy"])
    verdict: dict[str, Any] = {
        "accepted": False,
        "new_gain": round(new_gain, 6),
        "ece_delta_new": round(after_new["ece"] - before_new["ece"], 6),
        "min_new_gain": min_new_gain,
        "max_old_drop": max_old_drop,
        "old_drop": None,
        "ece_delta_old": None,
        "reasons": [],
    }
    if new_gain < min_new_gain:
        verdict["reasons"].append(
            f"new domain gain {new_gain:+.4f} < {min_new_gain:+.4f}"
        )
    if before_old is not None and after_old is not None:
        old_drop = float(before_old["accuracy"] - after_old["accuracy"])
        verdict["old_drop"] = round(old_drop, 6)
        verdict["ece_delta_old"] = round(after_old["ece"] - before_old["ece"], 6)
        if old_drop > max_old_drop:
            verdict["reasons"].append(
                f"old domain drop {old_drop:.4f} > {max_old_drop:.4f}"
            )
    verdict["accepted"] = not verdict["reasons"]
    return verdict


def load_pack(pack: str | Path) -> tuple[DecisionHeads, dict[str, Any]]:
    """Головы и ``metadata.json`` из каталога пака (heads.pt + metadata.json)."""
    pack = Path(pack)
    checkpoint = pack / "heads.pt"
    if not checkpoint.is_file():
        raise ValueError(f"no heads.pt in {pack}")
    heads = DecisionHeads.load_checkpoint(checkpoint)
    metadata_path = pack / "metadata.json"
    metadata: dict[str, Any] = {}
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return heads, metadata


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _encoder_for(
    metadata: dict[str, Any], override: str | None = None, dim: int = 0
):
    """Энкодер пака: из metadata (``encoder`` + ``feature_dim``) или явный."""
    from training.recipe.train_heads import make_encoder

    name = override or str(metadata.get("encoder") or "hash")
    resolved_dim = dim or int(metadata.get("feature_dim") or 0)
    if not resolved_dim:
        raise ValueError("не знаю размерность: нет feature_dim в metadata, задайте --dim")
    return make_encoder(name, resolved_dim), name


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Retrain decision heads on feedback with replay and a gate"
    )
    parser.add_argument("--pack", type=Path, required=True, help="пак: heads.pt")
    parser.add_argument("--feedback", type=Path, required=True, help="JSONL с метками")
    parser.add_argument("--out", type=Path, required=True, help="куда писать новый пак")
    parser.add_argument("--encoder", default=None, help="по умолчанию — из metadata")
    parser.add_argument("--dim", type=int, default=0, help="размерность признаков")
    parser.add_argument(
        "--replay", type=float, default=0.5, help="доля старого dataset.jsonl в обучении"
    )
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--holdout", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-new-gain", type=float, default=DEFAULT_MIN_NEW_GAIN)
    parser.add_argument("--max-old-drop", type=float, default=DEFAULT_MAX_OLD_DROP)
    parser.add_argument(
        "--control",
        action="store_true",
        help="обучить контроль на replay без обратной связи и показать net-выигрыш",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="только измерить, пак не писать"
    )
    parser.add_argument("--report", type=Path, default=None)
    return parser


def main(
    argv: list[str] | None = None,
    *,
    encoder=None,
) -> dict[str, Any]:
    """CLI дообучения; ``encoder`` подменяет реальный энкодер в тестах."""
    args = _parser().parse_args(argv)
    heads, metadata = load_pack(args.pack)
    specs = dict(heads.questions)
    if encoder is None:
        encoder, encoder_name = _encoder_for(metadata, args.encoder, args.dim)
    else:
        encoder_name = type(encoder).__name__
    encoder = CachedEncoder(encoder)

    feedback = read_examples(args.feedback, specs)
    train_examples, holdout_examples = split_holdout(
        feedback, args.holdout, args.seed
    )

    old_examples: list[dict[str, Any]] = []
    dataset_path = Path(args.pack) / "dataset.jsonl"
    if dataset_path.is_file():
        old_examples = read_examples(dataset_path, specs)
    # Старый домен для retention: тех же примеров не учим, иначе метрика врёт.
    old_train, old_holdout = split_retention(old_examples, 0.2, args.seed + 7)
    replay_examples = sample_replay(old_train, args.replay, args.seed)

    before_new = summarize(
        evaluate_benchmark(specs, holdout_examples, encoder, heads)
    )
    before_old = (
        summarize(evaluate_benchmark(specs, old_holdout, encoder, heads))
        if old_holdout
        else None
    )
    control: dict[str, Any] | None = None
    if args.control and replay_examples:
        # Контроль: те же шаги на replay, но без свежих примеров. Разница
        # «после - контроль» и есть вклад обратной связи, а не дообучения.
        control_heads = retrain(
            heads,
            specs,
            encoder,
            [],
            replay_examples,
            steps=args.steps,
            lr=args.lr,
            batch_size=args.batch_size,
            seed=args.seed,
        )
        control = {
            "new": summarize(
                evaluate_benchmark(specs, holdout_examples, encoder, control_heads)
            ),
            "old": (
                summarize(evaluate_benchmark(specs, old_holdout, encoder, control_heads))
                if old_holdout
                else None
            ),
        }

    trained = retrain(
        heads,
        specs,
        encoder,
        train_examples,
        replay_examples,
        steps=args.steps,
        lr=args.lr,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    after_new = summarize(
        evaluate_benchmark(specs, holdout_examples, encoder, trained)
    )
    after_old = (
        summarize(evaluate_benchmark(specs, old_holdout, encoder, trained))
        if old_holdout
        else None
    )
    verdict = gate(
        before_new=before_new,
        after_new=after_new,
        before_old=before_old,
        after_old=after_old,
        min_new_gain=args.min_new_gain,
        max_old_drop=args.max_old_drop,
    )

    report: dict[str, Any] = {
        "pack": str(args.pack),
        "feedback": str(args.feedback),
        "feedback_sha256": _sha256(Path(args.feedback)),
        "encoder": encoder_name,
        "specs": sorted(specs),
        "n_feedback": len(feedback),
        "n_train": len(train_examples),
        "n_holdout": len(holdout_examples),
        "n_replay": len(replay_examples),
        "n_old_holdout": len(old_holdout),
        "replay": args.replay,
        "steps": args.steps,
        "lr": args.lr,
        "seed": args.seed,
        "before": {"new": before_new, "old": before_old},
        "after": {"new": after_new, "old": after_old},
        "control": control,
        "net": (
            {
                "new_gain": round(after_new["accuracy"] - control["new"]["accuracy"], 6),
                "old_delta": (
                    round(after_old["accuracy"] - control["old"]["accuracy"], 6)
                    if after_old and control["old"]
                    else None
                ),
            }
            if control
            else None
        ),
        "gate": verdict,
        "written": False,
    }

    if verdict["accepted"] and not args.dry_run:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        trained.save_checkpoint(out / "heads.pt")
        updated = dict(metadata)
        updated["feedback"] = {
            "source": str(Path(args.feedback).name),
            "sha256": report["feedback_sha256"],
            "n": len(feedback),
            "n_train": len(train_examples),
            "n_holdout": len(holdout_examples),
            "n_replay": len(replay_examples),
            "replay": args.replay,
            "steps": args.steps,
            "lr": args.lr,
            "seed": args.seed,
            "gate": verdict,
            "control": control,
            "net": report["net"],
            "temperature_recalibrated": False,
        }
        (out / "metadata.json").write_text(
            json.dumps(updated, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        # Старый датасет + принятая обратная связь: следующий replay увидит новый
        # домен, и головы не придётся переучивать с нуля.
        merged = old_examples + train_examples
        with (out / "dataset.jsonl").open("w", encoding="utf-8") as fh:
            for example in merged:
                fh.write(json.dumps(example, ensure_ascii=False) + "\n")
        report["written"] = True

    report_path = args.report or Path(args.out) / "feedback.report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    status = "accepted" if verdict["accepted"] else "rejected"
    print(
        f"feedback {status}: new {before_new['accuracy']:.4f} -> "
        f"{after_new['accuracy']:.4f}"
        + (
            f", old {before_old['accuracy']:.4f} -> {after_old['accuracy']:.4f}"
            if before_old and after_old
            else ""
        ),
        flush=True,
    )
    if not verdict["accepted"]:
        for reason in verdict["reasons"]:
            print(f"  reject: {reason}", flush=True)
    print(f"report -> {report_path}", flush=True)
    return report


if __name__ == "__main__":
    main()
    sys.exit(0)
