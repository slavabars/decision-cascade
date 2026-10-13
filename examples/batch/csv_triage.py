"""CSV -> CSV: готовый конвейер триажа выгрузки обращений.

Читает CSV с колонкой текста, прогоняет все строки одним батчем in-process и
дописывает колонки с решениями. Если ``--input`` не указан, генерируется
демонстрационная выгрузка — пример запускается «из коробки».

Запуск:

    .venv/bin/python examples/batch/csv_triage.py --n 20
    .venv/bin/python examples/batch/csv_triage.py --input tickets.csv --output out.csv

Формат входа: колонка ``--text-column`` (по умолчанию ``text``) с текстом и
необязательный ``id``. Формат выхода: id, text + команда/срочность/эскалация/накал.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples._common import (
    build_service,
    primary_value,
    read_csv_rows,
    sample_tickets,
    ticket_specs,
    write_csv_rows,
)

OUTPUT_COLUMNS = ["id", "text", "команда", "срочность", "эскалация", "накал"]


def load_input(args: argparse.Namespace) -> list[dict[str, str]]:
    if args.input:
        rows = read_csv_rows(args.input, text_column=args.text_column)
        return [
            {"id": row.get("id") or str(i), "text": row[args.text_column]}
            for i, row in enumerate(rows)
        ]
    tickets = sample_tickets(args.n, seed=args.seed)
    return [{"id": t["id"], "text": t["state"]} for t in tickets]


def main() -> None:
    parser = argparse.ArgumentParser(description="CSV triage with decision-cascade API")
    parser.add_argument("--input", default=None, help="входной CSV (иначе — демо)")
    parser.add_argument("--output", default="data/examples/triage.csv")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--n", type=int, default=20, help="строк в демо-выгрузке")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rows = load_input(args)
    service = build_service(seed=args.seed)
    results = service.decide_many([row["text"] for row in rows], ticket_specs())

    out_rows = []
    for row, decisions in zip(rows, results):
        out_rows.append(
            {
                "id": row["id"],
                "text": row["text"],
                "команда": primary_value(decisions, "команда"),
                "срочность": round(float(primary_value(decisions, "срочность") or 0), 3),
                "эскалация": round(float(primary_value(decisions, "эскалация") or 0), 3),
                "накал": round(float(primary_value(decisions, "накал") or 0), 3),
            }
        )

    path = write_csv_rows(args.output, out_rows, OUTPUT_COLUMNS)
    print(f"Готово: {len(out_rows)} строк -> {path}")
    for row in out_rows[:5]:
        print(f"  {row['id']:<12} {row['команда']:<18} сроч={row['срочность']}")
    if len(out_rows) > 5:
        print(f"  … ещё {len(out_rows) - 5}")


if __name__ == "__main__":
    main()
