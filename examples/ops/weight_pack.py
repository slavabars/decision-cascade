"""Запуск с обученным паком весов (реальный энкодер, обученные головы).

Пак — это каталог ``checkpoints/packs/<dataset>-<model>/`` из релиза
(``support-gpt2``, ``public-gpt2`` и т.д.) и переменные ``DECISION_MODEL_ID`` /
``DECISION_FEATURE_DIM`` / ``DECISION_CHECKPOINT_DIR``. Здесь то же самое, но без
сервера: encoder + checkpoint_dir задаются явно.

Запуск:

    .venv/bin/python examples/ops/weight_pack.py
    .venv/bin/python examples/ops/weight_pack.py \\
        --pack checkpoints/packs/public-gpt2 --model openai-community/gpt2 --dim 768

Если пака нет — печатается подсказка, как его получить (скрипт завершается 0).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples._common import build_service, print_decisions, ticket_specs

STATE = "Не приходит выплата Stripe третьи сутки, теряем выручку!"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run with a trained weight pack")
    parser.add_argument("--pack", default="checkpoints/packs/support-gpt2")
    parser.add_argument("--model", default="openai-community/gpt2")
    parser.add_argument("--dim", type=int, default=768)
    parser.add_argument("--device", default=None, help="cpu / cuda / auto")
    parser.add_argument("--dtype", default=None, help="float32 / bfloat16 / float16")
    parser.add_argument("--kind", default=None, help="causal / encoder")
    args = parser.parse_args()

    pack = Path(args.pack)
    heads = sorted(pack.glob("*.pt")) if pack.exists() else []
    if not heads:
        print(f"Пак не найден: {pack}")
        print("Скачайте релиз (см. README) или обучите: training/recipe/train_weight_packs.sh")
        return

    service = build_service(
        checkpoint_dir=pack,
        model_id=args.model,
        feature_dim=args.dim,
        encoder_kind=args.kind,
        device=args.device,
        dtype=args.dtype,
    )
    decisions = service.decide(STATE, ticket_specs())
    print(f"Пак: {pack} (голов: {len(heads)}), модель: {args.model}")
    print_decisions(decisions)


if __name__ == "__main__":
    main()
