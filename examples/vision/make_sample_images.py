"""Генерирует синтетические изображения (фигуры) для остальных vision-примеров.

Запуск:

    .venv/bin/python examples/vision/make_sample_images.py --n 8
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from eval.datasets.image_dataset import generate_images, write_manifest

DEFAULT_DIR = Path("data/examples/vision/images")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate sample images")
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--out", default=str(DEFAULT_DIR))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out_dir = Path(args.out)
    manifest = generate_images(args.n, out_dir, seed=args.seed)
    manifest_path = write_manifest(out_dir / "manifest.jsonl", manifest)

    print(f"Сгенерировано изображений: {len(manifest)} -> {out_dir}")
    print(f"Манифест с метками: {manifest_path}")
    for item in manifest[:5]:
        labels = item["labels"]
        print(f"  {item['file']}: {labels['color']} {labels['shape']} x{labels['count']}")


if __name__ == "__main__":
    main()
