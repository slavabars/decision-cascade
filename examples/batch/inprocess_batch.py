"""Батч решений in-process: много обращений за один прогон энкодера.

Показывает ``DecisionService.decide_many`` — без HTTP и без загрузки моделей.
Так удобно прогонять выгрузки обращений целиком и считать распределение команд.

Запуск:

    .venv/bin/python examples/batch/inprocess_batch.py --n 8
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples._common import build_service, primary_value, sample_tickets, ticket_specs


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch decisions in-process")
    parser.add_argument("--n", type=int, default=8, help="число обращений")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    tickets = sample_tickets(args.n, seed=args.seed)
    states = [ticket["state"] for ticket in tickets]
    service = build_service(seed=args.seed)

    started = time.perf_counter()
    results = service.decide_many(states, ticket_specs())
    elapsed = time.perf_counter() - started

    print(f"Обработано обращений: {len(states)} за {elapsed:.3f} c")
    print(f"{'id':<12} {'команда':<18} {'сроч':>5} {'накал':>6} {'эскал':>6}")
    print("-" * 52)
    for ticket, decisions in zip(tickets, results):
        urgency = primary_value(decisions, "срочность")
        heat = primary_value(decisions, "накал")
        escalation = primary_value(decisions, "эскалация")
        team = primary_value(decisions, "команда")
        print(
            f"{ticket['id']:<12} {team!s:<18} "
            f"{float(urgency or 0.0):>5.2f} {float(heat or 0.0):>6.2f} "
            f"{float(escalation or 0.0):>6.2f}"
        )


if __name__ == "__main__":
    main()
