"""Генератор синтетических изображений (фигуры) с метками для мультимодальных тестов."""

from __future__ import annotations

import json
from pathlib import Path
from random import Random
from typing import Any

from PIL import Image, ImageDraw

COLORS = {
    "red": (220, 50, 50),
    "green": (50, 180, 90),
    "blue": (60, 100, 220),
    "yellow": (230, 200, 40),
}
SHAPES = ("circle", "square", "triangle")
BACKGROUND = (245, 245, 245)
SIZE = 64


def _draw_shape(
    draw: ImageDraw.ImageDraw, shape: str, box: tuple[int, int, int, int], fill
):
    if shape == "circle":
        draw.ellipse(box, fill=fill)
    elif shape == "square":
        draw.rectangle(box, fill=fill)
    else:
        x0, y0, x1, y1 = box
        draw.polygon([(x0, y1), (x1, y1), ((x0 + x1) // 2, y0)], fill=fill)


def generate_images(n: int, out_dir: str | Path, seed: int = 0) -> list[dict[str, Any]]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = Random(seed)
    colors = list(COLORS)
    manifest: list[dict[str, Any]] = []
    for i in range(n):
        color = rng.choice(colors)
        shape = rng.choice(SHAPES)
        count = rng.randint(1, 3)
        image = Image.new("RGB", (SIZE, SIZE), BACKGROUND)
        draw = ImageDraw.Draw(image)
        margin = 6
        slot = (SIZE - 2 * margin) // count
        for k in range(count):
            x0 = margin + k * slot + 2
            x1 = margin + (k + 1) * slot - 2
            _draw_shape(
                draw, shape, (x0, margin + 8, x1, SIZE - margin - 8), COLORS[color]
            )
        name = f"img_{i:05d}.png"
        image.save(out_dir / name)
        manifest.append(
            {
                "file": name,
                "labels": {"color": color, "shape": shape, "count": count},
            }
        )
    return manifest


def write_manifest(path: str | Path, manifest: list[dict[str, Any]]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for item in manifest:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")
    return path
