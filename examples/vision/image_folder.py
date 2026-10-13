"""Папка изображений -> решения по каждому + агрегат по папке.

Пакетная обработка мультимодальных входов in-process (HashVisionEncoder).
Удобно как заготовка для разбора вложений из тикетов.

Запуск:

    .venv/bin/python examples/vision/make_sample_images.py --n 8
    .venv/bin/python examples/vision/image_folder.py
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples._common import (
    build_service,
    primary_value,
    print_decisions,
    specs_from_api,
)

DEFAULT_DIR = Path("data/examples/vision/images")

QUESTIONS = {
    "тип": {
        "type": "choice",
        "instructions": "Что изображено",
        "options": ["фигура", "график", "текст", "скриншот ошибки"],
    },
    "критично": {
        "type": "noul",
        "instructions": "Изображение показывает критичную ошибку",
    },
}

STATE = "Вложение из обращения в поддержку."


def main() -> None:
    parser = argparse.ArgumentParser(description="Folder of images -> decisions")
    parser.add_argument("--dir", default=str(DEFAULT_DIR))
    parser.add_argument("--weight", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--show-first", type=int, default=1)
    args = parser.parse_args()

    folder = Path(args.dir)
    images = sorted(folder.glob("*.png"))
    if not images:
        from eval.datasets.image_dataset import generate_images

        generate_images(6, folder, seed=args.seed)
        images = sorted(folder.glob("*.png"))

    service = build_service(vision=True)
    specs = specs_from_api(QUESTIONS)
    types: Counter[str] = Counter()
    critical = 0

    for index, image in enumerate(images):
        decisions = service.decide(
            STATE, specs, images=[str(image)], media_kwargs={"media_weight": args.weight}
        )
        kind = primary_value(decisions, "тип")
        is_critical = float(primary_value(decisions, "критично") or 0.0) >= 0.5
        types[kind] += 1
        critical += int(is_critical)
        if index < args.show_first:
            print(f"\n# {image.name}")
            print_decisions(decisions)

    print(f"\nОбработано изображений: {len(images)}")
    print(f"Распределение по типу: {dict(types)}")
    print(f"Помечено критичными: {critical}")


if __name__ == "__main__":
    main()
