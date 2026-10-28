"""Подбор температуры логитов на отложенном val (temperature scaling).

Головы решений часто переуверены: `escalate`-голова на RouterBench выдаёт
0/1 при точности 0.78, из-за чего ECE равен Brier и каскад эскалирует не туда.
Температура ``T`` делит логиты перед softmax/sigmoid и **не меняет** argmax,
поэтому точность остаётся прежней, а вероятности становятся честными.

``T`` подбирается по NLL на **val** (тот же срез, что использует
``calibrate_rlcd``), а метрики до/после печатаются на val и, если он есть, на
test — чтобы видеть, что калибровка не подгоняется под срез.

Пример:
    python -m training.recipe.calibrate_temperature --specs routerbench \
        --encoder openai-community/gpt2 --dim 768 --limit 4000 \
        --checkpoint-dir checkpoints/packs/routerbench-gpt2 --device cuda \
        --out checkpoints/packs/routerbench-gpt2
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from training.recipe.train_heads import _make_encoder

# Сетка по T: 1.0..1000 лог-равномерно. Шаг ~4% — точнее, чем нужно для
# отчёта, и считается мгновенно (один forward на вопрос).
#
# Нижняя граница — 1.0: калибровка **только смягчает**. NLL не ограничена снизу
# при идеальном разделении классов, поэтому на маленьком val (200 примеров,
# acc=1.0) оптимум уходит в T→0 и превращает случайное совпадение в
# уверенность 1.0 — это худший из возможных сбоев для движка решений.
# Верхняя граница широкая намеренно: на RouterBench точность `route` почти
# равна случаю, и оптимум уходит далеко вправо — обрезать сетку на 20 значило
# бы вернуть переуверенные вероятности.
T_MIN, T_MAX, T_STEPS = 1.0, 1000.0, 200


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Temperature scaling of heads")
    parser.add_argument(
        "--specs", choices=["support", "public", "routerbench"], default="support"
    )
    parser.add_argument(
        "--shot",
        choices=["0shot", "5shot"],
        default="0shot",
        help="RouterBench: версия промптов (0-shot или 5-shot)",
    )
    parser.add_argument("--encoder", default="hash", help="'hash' или model_id")
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--limit", type=int, default=0, help="размер датасета")
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="куда сохранить головы с температурой (по умолчанию — в --checkpoint-dir)",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser


def _load_splits(args):
    """train/val/test в том же разбиении, что у train_heads и calibrate_rlcd."""
    from eval.datasets.text_dataset import (
        QUESTIONS,
        generate_examples,
        spec_from_api,
        split_examples,
    )

    if args.specs == "public":
        from eval.datasets.public_dataset import PUBLIC_SPECS, load_ag_news

        base = load_ag_news(args.limit or None)
        carved = split_examples(
            base["train"], val_fraction=0.2, test_fraction=0.0, seed=args.seed
        )
        return PUBLIC_SPECS, carved["train"], carved["val"], base["test"]

    if args.specs == "routerbench":
        from eval.datasets.routerbench import ROUTERBENCH_QUESTIONS, load_routerbench

        base = load_routerbench(args.limit or None, shot=args.shot)
        carved = split_examples(
            base["train"], val_fraction=0.2, test_fraction=0.0, seed=args.seed
        )
        specs = {
            name: spec_from_api(data) for name, data in ROUTERBENCH_QUESTIONS.items()
        }
        return specs, carved["train"], carved["val"], base["test"]

    specs = {name: spec_from_api(data) for name, data in QUESTIONS.items()}
    examples = generate_examples(n=args.limit or 1000, seed=args.seed)
    splits = split_examples(
        examples, val_fraction=0.2, test_fraction=0.2, seed=args.seed
    )
    return specs, splits["train"], splits["val"], splits["test"]


def _grid() -> list[float]:
    log_min, log_max = torch.log(torch.tensor(T_MIN)), torch.log(torch.tensor(T_MAX))
    steps = torch.linspace(float(log_min), float(log_max), T_STEPS)
    return [float(t) for t in torch.exp(steps)]


def fit_temperature(logits: torch.Tensor, targets: torch.Tensor, *, noul: bool) -> float:
    """``T``, минимизирующая NLL на переданном срезе (``logits`` — «сырые»)."""
    best_t, best_loss = 1.0, float("inf")
    for t in _grid():
        scaled = logits / t
        if noul:
            loss = float(F.binary_cross_entropy_with_logits(scaled, targets))
        else:
            loss = float(F.cross_entropy(scaled, targets))
        if loss < best_loss:
            best_t, best_loss = t, loss
    return best_t


def _report(specs, examples, features, heads) -> dict:
    from eval.benchmark.core import _metrics_report

    return _metrics_report(specs, examples, features, heads)


def _accept(before: dict, after: dict) -> bool:
    """Гейт «не навреди»: ECE на val должен улучшиться, Brier — не ухудшиться.

    NLL-оптимум температуры иногда смягчает голову, которая уже была
    откалибрована (типичный случай — бинарный `noul` с малой долей класса):
    Brier выигрывает, а ECE по вероятности положительного класса растёт.
    Такую температуру не применяем — оставляем ``T = 1.0``.
    """
    return after["ece"] < before["ece"] and after["brier"] <= before["brier"]


def main(argv: list[str] | None = None) -> dict:
    from eval.benchmark.core import _build_targets, encode_examples
    from src.decision_head.heads import DecisionHeads, build_option_embeddings
    from src.sampler.questions import NoulSpec, questions_signature

    args = _parser().parse_args(argv)
    torch.manual_seed(args.seed)

    specs, _train, val, test = _load_splits(args)
    encoder = _make_encoder(args.encoder, args.dim)
    heads = DecisionHeads.load_checkpoint(
        Path(args.checkpoint_dir) / "heads.pt",
        seed=args.seed,
        option_embeddings=build_option_embeddings(specs, encoder.encode),
    )
    # Подбор всегда идёт «с нуля»: если в чекпоинте уже была температура, она
    # не должна попадать в замер «до» (иначе повторный прогон несравним).
    heads.set_temperatures({name: 1.0 for name in specs})

    val_features = encode_examples(specs, val, encoder)
    test_features = encode_examples(specs, test, encoder) if test else {}
    val_targets = {
        name: _build_targets(spec, [ex["labels"][name] for ex in val])
        for name, spec in specs.items()
    }

    before_val = _report(specs, val, val_features, heads)
    before_test = _report(specs, test, test_features, heads) if test else None

    fitted: dict[str, float] = {}
    for name, spec in specs.items():
        with torch.no_grad():
            logits = heads.heads[name].logits(val_features[name])
        fitted[name] = fit_temperature(
            logits, val_targets[name], noul=isinstance(spec, NoulSpec)
        )
    heads.set_temperatures(fitted)

    after_val = _report(specs, val, val_features, heads)
    # Если подбор и так дал T = 1.0, отклонять нечего — метрики не менялись.
    rejected = [
        name
        for name in specs
        if fitted[name] != 1.0
        and not _accept(before_val["questions"][name], after_val["questions"][name])
    ]
    if rejected:
        heads.set_temperatures({name: 1.0 for name in rejected})
        after_val = _report(specs, val, val_features, heads)

    after_test = _report(specs, test, test_features, heads) if test else None

    applied = {name: 1.0 if name in rejected else t for name, t in fitted.items()}

    print(
        f"Температуры (specs={args.specs}, encoder={args.encoder}, "
        f"fit на val n={len(val)})"
    )
    for name, t in fitted.items():
        b0, b1 = before_val["questions"][name], after_val["questions"][name]
        mark = "  (отклонена гейтом)" if name in rejected else ""
        print(
            f"  {name:<12} T={t:5.2f}  val brier {b0['brier']:.4f}->{b1['brier']:.4f}"
            f"  ece {b0['ece']:.4f}->{b1['ece']:.4f}"
            f"  acc {b0['accuracy']:.3f}->{b1['accuracy']:.3f}{mark}"
        )
    print(
        f"  overall val acc {before_val['overall_accuracy']:.4f}->"
        f"{after_val['overall_accuracy']:.4f}"
    )
    if before_test is not None:
        print(
            f"  overall test acc {before_test['overall_accuracy']:.4f}->"
            f"{after_test['overall_accuracy']:.4f}"
        )
        for name in specs:
            b0, b1 = before_test["questions"][name], after_test["questions"][name]
            print(
                f"  {name:<12} test brier {b0['brier']:.4f}->{b1['brier']:.4f}"
                f"  ece {b0['ece']:.4f}->{b1['ece']:.4f}"
            )

    out_dir = Path(args.out or args.checkpoint_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    heads.save_checkpoint(out_dir / "heads.pt")
    heads.save_checkpoint(out_dir / f"{questions_signature(specs)}.pt")
    summary = {
        "specs": args.specs,
        "encoder": args.encoder,
        "feature_dim": heads.dim,
        "seed": args.seed,
        "temperature": applied,
        "fitted": fitted,
        "rejected": rejected,
        "before": {"val": before_val, "test": before_test},
        "after": {"val": after_val, "test": after_test},
    }
    (out_dir / "temperature.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _refresh_metadata(out_dir, applied, after_val, after_test, summary)
    print(f"  saved -> {out_dir}")
    return {"temperature": applied, "before": before_val, "after": after_val}


def _refresh_metadata(
    out_dir: Path,
    fitted: dict[str, float],
    after_val: dict,
    after_test: dict | None,
    summary: dict,
) -> None:
    """Обновляет ``metadata.json`` пака: иначе в нём останутся ECE «до».

    ``train``-метрики не пересчитываются (температура подбиралась на val), но
    ``val``/``test`` заменяются откалиброванными — чтобы метаданные описывали
    тот чекпоинт, который лежит рядом.
    """
    path = out_dir / "metadata.json"
    if not path.exists():
        return
    meta = json.loads(path.read_text(encoding="utf-8"))
    meta["temperature"] = fitted
    metrics = meta.setdefault("metrics", {})
    metrics["val"] = after_val
    if after_test is not None:
        metrics["test"] = after_test
    meta["calibration"] = {
        "fit": "val",
        "metric": "nll",
        "before": summary["before"],
        "after": summary["after"],
    }
    path.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
