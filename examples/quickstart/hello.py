"""Самый быстрый вход: решения in-process, без сервера и без загрузок.

Запуск:

    .venv/bin/python examples/quickstart/hello.py

Сервис по умолчанию использует детерминированный hash-энкодер и необученные
головы: это демонстрация конвейера, а не «смысл» чисел. Для реального качества
скачайте пак весов (см. ``examples/ops/weight_pack.py``) или обучите свои головы
(``training/``).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples._common import build_service, print_decisions, ticket_specs

STATE = (
    "Здравствуйте! Уже третьи сутки не можем подключить Stripe: при оплате "
    "клиенты видят «400: Invalid account». Пересоздание ключей не помогло, "
    "за это время потеряли около 15 заказов. Если к пятнице не решите — уйдём "
    "к конкурентам."
)


def main() -> None:
    print("=== decision-cascade: hello ===\n")
    print(f"Состояние:\n    {STATE}\n")

    service = build_service(seed=0)
    decisions = service.decide(STATE, ticket_specs())

    print(f"Ответ по {len(decisions)} вопросам (noul / choice / score):\n")
    print("-" * 52)
    print_decisions(decisions)
    print("-" * 52)
    print("\nПодсказка: то же самое через HTTP — examples/quickstart/quickstart.sh")


if __name__ == "__main__":
    main()
