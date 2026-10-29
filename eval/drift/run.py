"""Дрейф домена: сколько обратной связи нужно, чтобы вернуть качество.

Сценарий на AG News, где домен задаётся темами:

1. головы учатся только на **старом** срезе (по умолчанию World + Sports);
2. измеряется качество на старом test и на **новом** (Business + Sci/Tech);
3. из нового train берутся ``K`` примеров «обратной связи», головы дообучаются
   с replay старого датасета;
4. снова измеряется новый test (выигрыш) и старый (retention);
5. печатается таблица ``K -> accuracy/ECE`` и пишется JSON с сырыми числами.

Так получается измеренный ответ на вопрос «домен дрейфует — что делать»:
не «дообучайте на всём», а «столько-то примеров с replay дают столько-то п.п.».

Пример:
    python -m eval.drift.run --encoder hash --dim 256 --limit 2000 \
        --sizes 50,200,1000 --out data/drift/hash.json

    DECISION_DEVICE=cuda python -m eval.drift.run \
        --encoder openai-community/gpt2 --dim 768 --limit 4000 \
        --sizes 50,200,1000 --out data/drift/gpt2.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from random import Random
from typing import Any

from eval.benchmark.core import build_targets, encode_examples, evaluate_benchmark
from eval.datasets.public_dataset import (
    AG_NEWS_LABELS,
    AG_NEWS_REVISION,
    PUBLIC_SPECS,
    load_ag_news,
)
from src.decision_head.heads import DecisionHeads
from src.encoder.caching import CachedEncoder
from training.feedback.run import (
    DEFAULT_MAX_OLD_DROP,
    DEFAULT_MIN_NEW_GAIN,
    gate,
    retrain,
    sample_replay,
    summarize,
)
from training.recipe.train_heads import make_encoder
from training.sft.train import SFTTrainer


def topic_index(example: dict[str, Any]) -> int:
    """Индекс темы AG News из размеченного примера."""
    return AG_NEWS_LABELS.index(example["labels"]["topic"])


def split_by_topic(
    examples: list[dict[str, Any]], topics: set[int]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Делит набор на «в темах» и «вне тем»."""
    inside = [example for example in examples if topic_index(example) in topics]
    outside = [example for example in examples if topic_index(example) not in topics]
    return inside, outside


def sample_by_prior(
    examples: list[dict[str, Any]],
    prior: list[float],
    n: int,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Выборка с заданным распределением тем (имитация сдвига приора).

    ``prior`` — доли тем в порядке ``AG_NEWS_LABELS``; недостающие примеры темы
    берутся по остатку, а не выдумываются, поэтому размер может быть меньше ``n``.
    """
    by_topic: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for example in examples:
        by_topic[topic_index(example)].append(example)
    rng = Random(seed)
    picked: list[dict[str, Any]] = []
    for topic, share in enumerate(prior):
        if share <= 0:
            continue
        pool = list(by_topic.get(topic, []))
        rng.shuffle(pool)
        picked.extend(pool[: round(n * share)])
    return picked


def train_base(
    examples: list[dict[str, Any]],
    encoder,
    *,
    steps: int,
    lr: float,
    seed: int,
) -> DecisionHeads:
    """Головы, обученные только на старом домене."""
    features = encode_examples(PUBLIC_SPECS, examples, encoder)
    dim = next(iter(features.values())).shape[-1]
    heads = DecisionHeads(dim, PUBLIC_SPECS, seed=seed)
    targets = {
        name: build_targets(spec, [ex["labels"][name] for ex in examples])
        for name, spec in PUBLIC_SPECS.items()
    }
    SFTTrainer(heads, lr=lr).fit(features, targets, steps=steps)
    return heads


def _mean_scores(scores: list[dict[str, float]]) -> dict[str, float]:
    return {
        key: float(statistics.fmean(score[key] for score in scores))
        for key in scores[0]
    }


def _std_scores(scores: list[dict[str, float]]) -> dict[str, float]:
    if len(scores) < 2:
        return {key: 0.0 for key in scores[0]}
    return {
        key: float(statistics.stdev(score[key] for score in scores))
        for key in scores[0]
    }


def measure(examples: list[dict[str, Any]], encoder, heads: DecisionHeads) -> dict:
    if not examples:
        return {"accuracy": float("nan"), "ece": float("nan")}
    return summarize(evaluate_benchmark(PUBLIC_SPECS, examples, encoder, heads))


def run_experiment(
    *,
    limit: int,
    old_topics: set[int],
    sizes: list[int],
    mode: str = "label-shift",
    prior: list[float] | None = None,
    encoder,
    encoder_name: str,
    train_steps: int = 300,
    steps: int = 200,
    lr: float = 0.1,
    replay: float = 1.0,
    repeats: int = 1,
    control: bool = False,
    seed: int = 0,
    min_new_gain: float = DEFAULT_MIN_NEW_GAIN,
    max_old_drop: float = DEFAULT_MAX_OLD_DROP,
) -> dict[str, Any]:
    """Один прогон: базовые головы, затем дообучение на K примерах нового домена."""
    if prior is None:
        prior = [0.45, 0.45, 0.05, 0.05]
    if mode not in ("label-shift", "new-topics"):
        raise ValueError(f"unknown mode {mode!r}; expected label-shift or new-topics")
    splits = load_ag_news(limit)
    if mode == "new-topics":
        old_train, new_train = split_by_topic(splits["train"], old_topics)
        old_test, new_test = split_by_topic(splits["test"], old_topics)
    else:
        # Сдвиг приора: классы те же, меняется их доля во времени.
        new_prior = [round(1.0 - share, 4) for share in prior]
        old_train = sample_by_prior(splits["train"], prior, limit, seed)
        new_train = sample_by_prior(splits["train"], new_prior, limit, seed + 1)
        old_test = sample_by_prior(splits["test"], prior, len(splits["test"]), seed + 2)
        new_test = sample_by_prior(
            splits["test"], new_prior, len(splits["test"]), seed + 3
        )
    if not old_train or not new_train or not new_test:
        raise ValueError(
            "не хватает данных: проверьте --old-topics/--prior и --limit "
            f"(old_train={len(old_train)}, new_train={len(new_train)}, "
            f"new_test={len(new_test)})"
        )

    base = train_base(old_train, encoder, steps=train_steps, lr=lr, seed=seed)
    baseline = {
        "old": measure(old_test, encoder, base),
        "new": measure(new_test, encoder, base),
    }

    if repeats < 1:
        raise ValueError(f"repeats must be >= 1, got {repeats}")

    shuffled = list(new_train)
    Random(seed + 3).shuffle(shuffled)
    runs: list[dict[str, Any]] = []
    if control:
        # Контроль: те же шаги и replay, но без свежих примеров. Если прирост
        # есть и здесь, его даёт дополнительное обучение, а не адаптация.
        replay_examples = sample_replay(old_train, replay, seed)
        trained = retrain(
            base,
            PUBLIC_SPECS,
            encoder,
            [],
            replay_examples,
            steps=steps,
            lr=lr,
            seed=seed,
        )
        after_new = measure(new_test, encoder, trained)
        after_old = measure(old_test, encoder, trained)
        runs.append(
            {
                "k": 0,
                "replay": replay,
                "repeats": 1,
                "new": after_new,
                "old": after_old,
                "new_std": {"accuracy": 0.0, "ece": 0.0},
                "old_std": {"accuracy": 0.0, "ece": 0.0},
                "gate": gate(
                    before_new=baseline["new"],
                    after_new=after_new,
                    before_old=baseline["old"],
                    after_old=after_old,
                    min_new_gain=min_new_gain,
                    max_old_drop=max_old_drop,
                ),
            }
        )
    for k in sizes:
        k = min(k, len(shuffled))
        new_scores: list[dict[str, float]] = []
        old_scores: list[dict[str, float]] = []
        for repeat in range(repeats):
            # Непересекающиеся куски: иначе «разные» прогоны делили бы примеры.
            start = repeat * k
            if start + k > len(shuffled):
                start = (repeat * k) % max(1, len(shuffled) - k + 1)
            feedback = shuffled[start : start + k]
            replay_examples = sample_replay(old_train, replay, seed + repeat)
            trained = retrain(
                base,
                PUBLIC_SPECS,
                encoder,
                feedback,
                replay_examples,
                steps=steps,
                lr=lr,
                seed=seed + repeat,
            )
            new_scores.append(measure(new_test, encoder, trained))
            old_scores.append(measure(old_test, encoder, trained))

        after_new = _mean_scores(new_scores)
        after_old = _mean_scores(old_scores)
        verdict = gate(
            before_new=baseline["new"],
            after_new=after_new,
            before_old=baseline["old"],
            after_old=after_old,
            min_new_gain=min_new_gain,
            max_old_drop=max_old_drop,
        )
        runs.append(
            {
                "k": k,
                "replay": replay,
                "repeats": repeats,
                "new": after_new,
                "old": after_old,
                "new_std": _std_scores(new_scores),
                "old_std": _std_scores(old_scores),
                "gate": verdict,
            }
        )

    return {
        "dataset": "ag_news",
        "mode": mode,
        "revision": AG_NEWS_REVISION,
        "encoder": encoder_name,
        "prior": prior,
        "old_topics": [AG_NEWS_LABELS[index] for index in sorted(old_topics)],
        "new_topics": [
            name for index, name in enumerate(AG_NEWS_LABELS) if index not in old_topics
        ],
        "limit": limit,
        "train_steps": train_steps,
        "steps": steps,
        "lr": lr,
        "repeats": repeats,
        "control": control,
        "seed": seed,
        "n": {
            "old_train": len(old_train),
            "old_test": len(old_test),
            "new_train": len(new_train),
            "new_test": len(new_test),
        },
        "baseline": baseline,
        "runs": runs,
    }


def render_markdown(report: dict[str, Any]) -> str:
    """Таблица для docs/drift.md: K, выигрыш на новом, retention на старом."""
    baseline = report["baseline"]
    lines = [
        f"# Дрейф домена: {report['dataset']} ({report['encoder']})",
        "",
        (
            f"Режим: {report['mode']}; старый домен: "
            f"{', '.join(report['old_topics'])}; новый: "
            f"{', '.join(report['new_topics'])}."
        ),
        "",
        "| K | новый acc | Δ новый | новый ECE | старый acc | Δ старый | гейт |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for run in report["runs"]:
        new_gain = run["new"]["accuracy"] - baseline["new"]["accuracy"]
        old_delta = run["old"]["accuracy"] - baseline["old"]["accuracy"]
        verdict = "принят" if run["gate"]["accepted"] else "отклонён"
        lines.append(
            f"| {run['k']} | {run['new']['accuracy']:.4f} | {new_gain:+.4f} | "
            f"{run['new']['ece']:.4f} | {run['old']['accuracy']:.4f} | "
            f"{old_delta:+.4f} | {verdict} |"
        )
    return "\n".join(lines) + "\n"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Domain drift: old slice -> new slice with feedback"
    )
    parser.add_argument("--encoder", default="hash", help="'hash' или model_id")
    parser.add_argument(
        "--mode",
        choices=["label-shift", "new-topics"],
        default="label-shift",
        help="label-shift — те же темы, другой приор; new-topics — новые темы",
    )
    parser.add_argument(
        "--prior",
        default="0.45,0.45,0.05,0.05",
        help="доли тем старого домена в порядке AG News; новый — дополнение до 1",
    )
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--limit", type=int, default=2000, help="примеров из train")
    parser.add_argument(
        "--old-topics",
        default="0,1",
        help="индексы тем AG News для старого домена (0 World, 1 Sports, 2 Business, 3 Sci/Tech)",
    )
    parser.add_argument("--sizes", default="50,200,1000", help="сколько примеров feedback")
    parser.add_argument("--train-steps", type=int, default=300)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--replay", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="сколько раз повторить дообучение (разные куски feedback) и усреднить",
    )
    parser.add_argument(
        "--control",
        action="store_true",
        help="добавить контроль: те же шаги на replay без обратной связи",
    )
    parser.add_argument("--min-new-gain", type=float, default=DEFAULT_MIN_NEW_GAIN)
    parser.add_argument("--max-old-drop", type=float, default=DEFAULT_MAX_OLD_DROP)
    parser.add_argument("--out", type=Path, default=None, help="JSON с сырыми числами")
    parser.add_argument(
        "--markdown", type=Path, default=None, help="куда сохранить таблицу"
    )
    return parser


def main(argv: list[str] | None = None, *, encoder=None) -> dict[str, Any]:
    args = _parser().parse_args(argv)
    old_topics = {int(part) for part in args.old_topics.split(",") if part.strip()}
    prior = [float(part) for part in args.prior.split(",") if part.strip()]
    if len(prior) != len(AG_NEWS_LABELS) or abs(sum(prior) - 1.0) > 1e-6:
        raise SystemExit(
            f"--prior должен задавать {len(AG_NEWS_LABELS)} долей с суммой 1.0"
        )
    sizes = [int(part) for part in args.sizes.split(",") if part.strip()]
    if encoder is None:
        encoder = make_encoder(args.encoder, args.dim)
    encoder = CachedEncoder(encoder)

    report = run_experiment(
        limit=args.limit,
        old_topics=old_topics,
        sizes=sizes,
        mode=args.mode,
        prior=prior,
        encoder=encoder,
        encoder_name=args.encoder,
        train_steps=args.train_steps,
        steps=args.steps,
        lr=args.lr,
        replay=args.replay,
        repeats=args.repeats,
        control=args.control,
        seed=args.seed,
        min_new_gain=args.min_new_gain,
        max_old_drop=args.max_old_drop,
    )
    markdown = render_markdown(report)
    print(markdown, flush=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"report -> {args.out}", flush=True)
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(markdown, encoding="utf-8")
        print(f"markdown -> {args.markdown}", flush=True)
    return report


if __name__ == "__main__":
    main()
    sys.exit(0)
