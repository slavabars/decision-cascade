"""Option-scoring голова: подмена набора вариантов выбора без переобучения.

Обычная голова выбора хранит **отдельный вес на каждый вариант** — набор вариантов
зашит на обучении. Option-scoring голова оценивает вариант общим скорером по его
текстовому эмбеддингу, поэтому одни и те же веса можно применить к другому набору
вариантов, просто пересчитав эмбеддинги.

Пример полностью офлайн: энкодер — hash (без весов). Обучаем голову на трёх
вариантах, затем подменяем набор на два невиданных и смотрим точность. На hash
переноса нет (мешок токенов не несёт семантики) — на реальном семантическом
энкодере результат заметно лучше, а честные числа и границы — в
`docs/option-scoring.md`.

Запуск:

    .venv/bin/python examples/ops/option_transfer.py
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from eval.benchmark.core import evaluate_benchmark, train_benchmark
from eval.option_transfer.core import (
    SEEN_OPTIONS,
    UNSEEN_OPTIONS,
    generate_examples,
    option_transfer_specs,
)
from eval.report.run import _encoder
from src.decision_head.heads import OptionScoringChoiceHead
from src.sampler.questions import ChoiceSpec, build_option_prompt, build_prompt

INSTRUCTIONS = "rate the meal"


def _accuracy(head, encoder, options, examples) -> float:
    spec = ChoiceSpec(INSTRUCTIONS, options)
    features = encoder.encode([build_prompt(e["state"], spec) for e in examples])
    with torch.no_grad():
        logits = head.logits(features)
    preds = [head.options[i] for i in logits.argmax(-1).tolist()]
    labels = [e["labels"]["polarity"] for e in examples]
    return sum(p == lab for p, lab in zip(preds, labels)) / len(labels)


def main() -> None:
    parser = argparse.ArgumentParser(description="Option-scoring transfer demo")
    parser.add_argument("--train-steps", type=int, default=800)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n-per-label", type=int, default=40)
    parser.add_argument("--encoder", default="hash", help="'hash' или model_id")
    parser.add_argument("--dim", type=int, default=64)
    args = parser.parse_args()

    dim = args.dim
    encoder = _encoder(args.encoder, dim, "auto", "float32")
    seen_spec = option_transfer_specs(SEEN_OPTIONS)

    train = generate_examples(SEEN_OPTIONS, args.n_per_label, 0)
    heads, _ = train_benchmark(
        seen_spec,
        train,
        encoder,
        train_steps=args.train_steps,
        lr=args.lr,
        seed=0,
        choice_head="option_scoring",
    )
    trained = heads.heads["polarity"]
    seen_acc = evaluate_benchmark(
        seen_spec, generate_examples(SEEN_OPTIONS, args.n_per_label, 1), encoder, heads
    )["questions"]["polarity"]["accuracy"]

    # Подмена набора вариантов: веса те же, меняются только эмбеддинги.
    embeddings = encoder.encode(
        [build_option_prompt(ChoiceSpec(INSTRUCTIONS, UNSEEN_OPTIONS), opt) for opt in UNSEEN_OPTIONS]
    )
    swapped = OptionScoringChoiceHead(dim, UNSEEN_OPTIONS, embeddings)
    swapped.load_state_dict(trained.state_dict())
    unseen_acc = _accuracy(
        swapped,
        encoder,
        UNSEEN_OPTIONS,
        generate_examples(UNSEEN_OPTIONS, args.n_per_label, 1),
    )

    print(f"обучение на вариантах: {list(SEEN_OPTIONS)}")
    print(f"  точность (те же варианты):        {seen_acc:.3f}\n")
    print(f"подмена набора на: {list(UNSEEN_OPTIONS)}  (те же веса)")
    print(f"  точность (невиданные варианты):   {unseen_acc:.3f}")
    print(f"  случайная точность:               {1 / len(UNSEEN_OPTIONS):.3f}")
    print()
    print(
        "Обычная (linear) голова так не умеет: её выходной слой имеет "
        f"{len(SEEN_OPTIONS)} выходов — по одному на обученный вариант."
    )
    if args.encoder == "hash":
        print(
            "\n[подсказка] hash-энкодер не несёт семантики — перенос на невиданные"
            "\nварианты тут остаётся на уровне случайной точности. Запустите с"
            "\nсемантической моделью, чтобы увидеть эффект:"
            "\n  .venv/bin/python examples/ops/option_transfer.py \\"
            "\n    --encoder openai-community/gpt2 --dim 768"
        )


if __name__ == "__main__":
    main()
