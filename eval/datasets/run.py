"""CLI: генерация синтетических датасетов (текст + изображения).

Пример:
    python -m eval.datasets.run --text-n 1000 --image-n 500 --out-dir data
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval.datasets.image_dataset import generate_images, write_manifest
from eval.datasets.text_dataset import QUESTIONS, generate_examples, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic datasets")
    parser.add_argument("--text-n", type=int, default=1000)
    parser.add_argument("--image-n", type=int, default=500)
    parser.add_argument("--out-dir", type=Path, default=Path("data"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    questions_path = args.out_dir / "questions.json"
    questions_path.write_text(
        json.dumps(QUESTIONS, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    text = generate_examples(args.text_n, seed=args.seed)
    text_path = write_jsonl(args.out_dir / "support_tickets.jsonl", text)

    images_dir = args.out_dir / "images"
    manifest = generate_images(args.image_n, images_dir, seed=args.seed)
    manifest_path = write_manifest(images_dir / "manifest.jsonl", manifest)

    print(f"text     : {len(text):>6} -> {text_path}")
    print(f"questions:         -> {questions_path}")
    print(f"images   : {len(manifest):>6} -> {images_dir}/ ({manifest_path.name})")


if __name__ == "__main__":
    main()
