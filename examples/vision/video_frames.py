"""Видео -> решения: поток разбивается на кадры (``max_frames``).

CLIP умеет только изображения, поэтому для видео берём ``HashVisionEncoder``
(офлайн). Здесь из нескольких картинок собирается псевдо-клип, чтобы показать
API: ``videos=[...]`` + ``media_kwargs={'max_frames': N}``.

Запуск:

    .venv/bin/python examples/vision/video_frames.py
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples._common import build_service, print_decisions, specs_from_api

IMAGES_DIR = Path("data/examples/vision/images")
CLIP_PATH = Path("data/examples/vision/clip.mp4")

QUESTIONS = {
    "тип": {
        "type": "choice",
        "instructions": "Что происходит в видео",
        "options": ["демонстрация", "ошибка", "обучение", "пусто"],
    },
    "критично": {"type": "noul", "instructions": "Видео показывает проблему"},
}

STATE = "Пользователь приложил короткую запись экрана."


def build_clip(folder: Path, clip_path: Path) -> Path:
    """Склеивает картинки в один файл — «сырьё» для кадров."""
    images = sorted(folder.glob("*.png"))
    if not images:
        from eval.datasets.image_dataset import generate_images

        generate_images(4, folder, seed=0)
        images = sorted(folder.glob("*.png"))
    clip_path.parent.mkdir(parents=True, exist_ok=True)
    with clip_path.open("wb") as out:
        for image in images:
            out.write(image.read_bytes())
    return clip_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Video frames decisions")
    parser.add_argument("--dir", default=str(IMAGES_DIR))
    parser.add_argument("--clip", default=str(CLIP_PATH))
    parser.add_argument("--frames", type=int, default=4)
    parser.add_argument("--weight", type=float, default=0.5)
    args = parser.parse_args()

    clip = build_clip(Path(args.dir), Path(args.clip))
    service = build_service(vision=True)
    decisions = service.decide(
        STATE,
        specs_from_api(QUESTIONS),
        videos=[str(clip)],
        media_kwargs={"max_frames": args.frames, "media_weight": args.weight},
    )

    print(f"Видео: {clip} (max_frames={args.frames})")
    print(f"Модальности сервиса: {sorted(service.modalities)}\n")
    print_decisions(decisions)


if __name__ == "__main__":
    main()
