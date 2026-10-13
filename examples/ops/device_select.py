"""Выбор устройства и типа вычислений: cpu / cuda / mps + dtype.

Показывает, как ``resolve_device``/``resolve_dtype`` разбирают окружение (та же
логика, что в сервере), и замеряет кодирование на выбранном устройстве. По
умолчанию берётся hash-энкодер — работает без GPU и без загрузок.

Запуск:

    .venv/bin/python examples/ops/device_select.py
    .venv/bin/python examples/ops/device_select.py --device cuda --dtype bfloat16
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples._common import sample_tickets
from src.device import check_dtype_device, resolve_device, resolve_dtype


def main() -> None:
    parser = argparse.ArgumentParser(description="Device/dtype selection")
    parser.add_argument("--device", default=None, help="auto / cpu / cuda / cuda:0 / mps")
    parser.add_argument("--dtype", default=None, help="float32 / float16 / bfloat16")
    parser.add_argument("--n", type=int, default=200)
    args = parser.parse_args()

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype)
    check_dtype_device(dtype, device)

    print(f"device={device}  dtype={dtype}")

    from api.server.service import HashStateEncoder

    encoder = HashStateEncoder(dim=64)
    states = [t["state"] for t in sample_tickets(args.n, seed=0)]
    started = time.perf_counter()
    features = encoder.encode(states)
    elapsed = time.perf_counter() - started

    print(f"encoded {len(states)} states -> {tuple(features.shape)} за {elapsed:.4f} c")
    print("Подсказка: для GPU-энкодера задайте --device cuda и DECISION_MODEL_ID.")


if __name__ == "__main__":
    main()
