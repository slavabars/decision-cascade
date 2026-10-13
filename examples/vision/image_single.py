"""Одно изображение -> решения: мультимодальный вход (текст + картинка).

По умолчанию используется ``HashVisionEncoder`` (офлайн, без загрузок): пример
запускается мгновенно. Для реального CLIP поднимите сервер с
``DECISION_VISION_MODEL_ID`` и обращайтесь по HTTP (см. README).

Запуск:

    .venv/bin/python examples/vision/image_single.py
    .venv/bin/python examples/vision/image_single.py --image shot.png --weight 0.8
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples._common import build_service, print_decisions, specs_from_api

DEFAULT_IMAGE = Path("data/examples/vision/images/img_00000.png")

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

STATE = "Во вложении скриншот проблемы из личного кабинета."


def ensure_image(path: Path) -> Path:
    """Возвращает картинку: если её нет — генерирует одну."""
    if path.exists():
        return path
    from eval.datasets.image_dataset import generate_images

    out_dir = path.parent
    generate_images(1, out_dir, seed=0)
    if not path.exists():
        alternatives = sorted(out_dir.glob("*.png"))
        if not alternatives:
            raise FileNotFoundError(f"no images in {out_dir}")
        return alternatives[0]
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Single image decisions")
    parser.add_argument("--image", default=str(DEFAULT_IMAGE))
    parser.add_argument("--weight", type=float, default=0.5, help="media_weight")
    args = parser.parse_args()

    image = ensure_image(Path(args.image))
    service = build_service(vision=True)
    decisions = service.decide(
        STATE,
        specs_from_api(QUESTIONS),
        images=[str(image)],
        media_kwargs={"media_weight": args.weight},
    )

    print(f"Изображение: {image} (media_weight={args.weight})")
    print(f"Модальности сервиса: {sorted(service.modalities)}\n")
    print_decisions(decisions)


if __name__ == "__main__":
    main()
