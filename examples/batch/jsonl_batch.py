"""JSONL -> JSONL через HTTP-эндпоинт ``/decisions/batch``.

Показывает серверный батч (а не in-process): один запрос — список состояний.
Требует запущенный сервер (см. ``examples/quickstart/quickstart.sh``).

Запуск:

    .venv/bin/python -m uvicorn api.server.main:app --port 8000
    .venv/bin/python examples/batch/jsonl_batch.py --n 10

Без ``--input`` генерируется демонстрационный JSONL; результат пишется в
``--output``. Формат строк входа: ``{"id": ..., "state": ...}``.
"""
import argparse
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples._common import (
    decision_to_dict,
    primary_value,
    read_jsonl,
    sample_tickets,
    ticket_questions,
    write_jsonl,
)

# Эндпоинт принимает максимум 256 состояний за запрос (см. api/server/main.py).
MAX_BATCH = 256


def load_states(args: argparse.Namespace) -> list[dict[str, str]]:
    if args.input:
        rows = read_jsonl(args.input)
        return [{"id": str(r.get("id")), "state": r["state"]} for r in rows]
    tickets = sample_tickets(args.n, seed=args.seed)
    return [{"id": t["id"], "state": t["state"]} for t in tickets]


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch decisions over HTTP")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--input", default=None, help="входной JSONL (иначе — демо)")
    parser.add_argument("--output", default="data/examples/batch.jsonl")
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rows = load_states(args)
    out_rows = []
    with httpx.Client(base_url=args.base_url, timeout=60.0) as client:
        for start in range(0, len(rows), MAX_BATCH):
            chunk = rows[start : start + MAX_BATCH]
            resp = client.post(
                "/decisions/batch",
                json={
                    "states": [row["state"] for row in chunk],
                    "questions": ticket_questions(),
                },
            )
            resp.raise_for_status()
            for row, decisions in zip(chunk, resp.json()["results"]):
                out_rows.append(
                    {
                        "id": row["id"],
                        "команда": primary_value(decisions, "команда"),
                        "срочность": primary_value(decisions, "срочность"),
                        "decisions": {
                            name: decision_to_dict(d)
                            for name, d in decisions.items()
                        },
                    }
                )

    path = write_jsonl(args.output, out_rows)
    print(f"Готово: {len(out_rows)} строк -> {path}")
    for row in out_rows[:5]:
        print(f"  {row['id']:<12} команда={row['команда']}")


if __name__ == "__main__":
    main()
