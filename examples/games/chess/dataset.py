"""Генерация данных для головы «нужен ли более глубокий просчёт».

Разметка не требует человека и не требует «правильных» шахматных ходов: метка
берётся из самого движка. Для каждой позиции делается быстрый просчёт (он же
даёт признаки и кандидатов) и глубокий; если глубокий нашёл другой ход или
оценка сдвинулась сильнее порога, быстрый ответ считается ненадёжным.

Позиции берутся из партий движка против самого себя: случайные ходы дают
неестественные тактические позиции и смещают метку.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import chess

from eval.datasets.chess import SPLIT_NAMES
from eval.datasets.text_dataset import split_examples
from examples.games.chess.engine import Candidate, UciEngine
from examples.games.chess.state import text_state

#: Сдвиг оценки, при котором быстрый ответ считается ненадёжным.
DEFAULT_DELTA_CP = 50


def is_unreliable(
    shallow: list[Candidate],
    deep: list[Candidate],
    *,
    delta_cp: int = DEFAULT_DELTA_CP,
) -> bool:
    """Расходится ли быстрый ответ с глубоким (ход или оценка)."""
    if not shallow or not deep:
        raise ValueError("both searches must return at least one candidate")
    if shallow[0].move != deep[0].move:
        return True
    return abs(deep[0].score - shallow[0].score) > delta_cp


def _first_plies(rng: random.Random, board: chess.Board, count: int) -> None:
    """Разнообразие дебютов: несколько случайных легальных ходов."""
    for _ in range(count):
        if board.is_game_over():
            return
        board.push(rng.choice(list(board.legal_moves)))


def generate(
    engine_path: str,
    *,
    games: int = 40,
    start_ply: int = 6,
    max_ply: int = 46,
    shallow_ms: int = 60,
    deep_ms: int = 260,
    delta_cp: int = DEFAULT_DELTA_CP,
    seed: int = 0,
    progress: bool = True,
) -> list[dict[str, Any]]:
    """Партии движка с самим собой -> примеры `state -> need_deep`."""
    rng = random.Random(seed)
    examples: list[dict[str, Any]] = []
    with UciEngine(engine_path, multipv=3, hash_mb=128) as engine:
        for game in range(games):
            board = chess.Board()
            _first_plies(rng, board, rng.choice((0, 1, 2)))
            while not board.is_game_over() and len(board.move_stack) < max_ply:
                fen = board.fen()
                shallow = engine.analyse(
                    fen, movetime_ms=shallow_ms, multipv=3
                )
                if len(board.move_stack) >= start_ply:
                    deep = engine.analyse(fen, movetime_ms=int(deep_ms), multipv=1)
                    label = is_unreliable(shallow, deep, delta_cp=delta_cp)
                    examples.append(
                        {
                            "id": f"g{game:03d}-p{len(board.move_stack):03d}",
                            "state": text_state(board, shallow),
                            "labels": {"need_deep": int(label)},
                            "fen": fen,
                            "shallow": shallow[0].describe(),
                            "deep": deep[0].describe(),
                            "delta_cp": abs(deep[0].score - shallow[0].score),
                        }
                    )
                    if progress and len(examples) % 100 == 0:
                        share = sum(e["labels"]["need_deep"] for e in examples)
                        print(
                            f"  {len(examples)} примеров, "
                            f"доля need_deep={share / len(examples):.2f}",
                            flush=True,
                        )
                board.push(chess.Move.from_uci(shallow[0].move))
            if progress:
                print(f"партия {game + 1}/{games}: {len(examples)} примеров", flush=True)
    return examples


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", required=True, help="путь к Stockfish")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/chess"),
        help="каталог для train/val/test.jsonl",
    )
    parser.add_argument("--games", type=int, default=40)
    parser.add_argument("--start-ply", type=int, default=6)
    parser.add_argument("--max-ply", type=int, default=46)
    parser.add_argument("--shallow-ms", type=int, default=60)
    parser.add_argument("--deep-ms", type=int, default=260)
    parser.add_argument("--delta-cp", type=int, default=DEFAULT_DELTA_CP)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    examples = generate(
        args.engine,
        games=args.games,
        start_ply=args.start_ply,
        max_ply=args.max_ply,
        shallow_ms=args.shallow_ms,
        deep_ms=args.deep_ms,
        delta_cp=args.delta_cp,
        seed=args.seed,
    )
    splits = split_examples(
        examples, val_fraction=0.2, test_fraction=0.2, seed=args.seed
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name in SPLIT_NAMES:
        path = args.out_dir / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for example in splits[name]:
                handle.write(json.dumps(example, ensure_ascii=False) + "\n")
        positives = sum(ex["labels"]["need_deep"] for ex in splits[name])
        share = positives / len(splits[name]) if splits[name] else 0.0
        print(f"-> {path}: {len(splits[name])} примеров, need_deep={share:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
