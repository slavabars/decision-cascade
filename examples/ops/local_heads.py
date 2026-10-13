"""In-process SDK без HTTP: решения в процессе + сохранение/загрузка голов.

Показывает, что сервер не обязателен: ``DecisionService`` вызывается напрямую, а
головы можно сохранить в чекпоинт и переиспользовать (в т.ч. между запусками).

Запуск:

    .venv/bin/python examples/ops/local_heads.py
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples._common import build_service, primary_value, ticket_specs

STATE = "Не приходит выплата Stripe третьи сутки, теряем выручку!"


def main() -> None:
    parser = argparse.ArgumentParser(description="In-process decisions + checkpoint")
    parser.add_argument("--dir", default="data/examples/heads")
    args = parser.parse_args()

    checkpoint_dir = Path(args.dir)
    service = build_service(seed=0, checkpoint_dir=checkpoint_dir)
    specs = ticket_specs()

    before = service.decide(STATE, specs)
    path = service.save_heads(service.dim, specs)
    print(f"Чекпоинт сохранён: {path}")

    reloaded = build_service(seed=999, checkpoint_dir=checkpoint_dir)
    after = reloaded.decide(STATE, specs)

    print(f"команда:  было {primary_value(before, 'команда')!r}, "
          f"стало {primary_value(after, 'команда')!r}")
    print(f"срочность: было {float(primary_value(before, 'срочность') or 0):.4f}, "
          f"стало {float(primary_value(after, 'срочность') or 0):.4f}")
    print("Совпадает: головы загружены из чекпоинта, а не пересозданы сидом.")


if __name__ == "__main__":
    main()
