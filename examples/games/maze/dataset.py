"""Разметка позиций: где жадный шаг ведёт к смерти.

Метка не придумана человеком и не скопирована у эвристики: мир прогоняется
вперёд на `horizon` шагов жадной политикой, и `need_deep = 1`, если герой за это
время теряет жизнь. Это ровно то событие, ради которого каскад платит за
осторожный маршрут, поэтому учить и измерять его можно на одном критерии.

Эпизоды для датасета играются смешанной политикой (жадная плюс случайные ходы),
иначе позиции похожи друг на друга и голова учится на одном и том же коридоре.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from examples.games.maze.env import DIRECTIONS, Maze, Pos
from examples.games.maze.planner import greedy_move
from examples.games.maze.state import text_state

DEFAULT_HORIZON = 12


def _greedy_dies(maze: Maze, first: Pos, *, horizon: int) -> bool:
    """Теряет ли герой жизнь в ближайшие `horizon` шагов жадной игры."""
    game = maze.copy()
    deaths = game.deaths
    move = first
    for _ in range(horizon):
        if game.done:
            return False
        game.step(move)
        if game.deaths > deaths:
            return True
        move, _ = greedy_move(game)
    return False


def label_position(maze: Maze, *, horizon: int = DEFAULT_HORIZON) -> dict[str, Any]:
    """Пример `state -> need_deep` для текущей позиции."""
    greedy, _ = greedy_move(maze)
    return {
        "need_deep": int(_greedy_dies(maze, greedy, horizon=horizon)),
        "greedy": greedy,
    }


def generate(
    *,
    episodes: int = 30,
    steps: int = 300,
    horizon: int = DEFAULT_HORIZON,
    stride: int = 3,
    epsilon: float = 0.15,
    seed: int = 0,
    progress: bool = True,
) -> list[dict[str, Any]]:
    """Партии смешанной политикой -> размеченные позиции."""
    rng = random.Random(seed)
    examples: list[dict[str, Any]] = []
    for episode in range(episodes):
        game = Maze(seed=seed * 1000 + episode, max_steps=steps)
        step = 0
        while not game.done:
            if step % stride == 0:
                labelled = label_position(game, horizon=horizon)
                examples.append(
                    {
                        "id": f"e{episode:03d}-s{step:03d}",
                        "state": text_state(game),
                        "labels": {"need_deep": labelled["need_deep"]},
                        "score": game.score,
                        "steps_left": steps - step,
                    }
                )
            if rng.random() < epsilon:
                move = rng.choice(DIRECTIONS)
            else:
                move, _ = greedy_move(game)
            game.step(move)
            step += 1
        if progress:
            positives = sum(ex["labels"]["need_deep"] for ex in examples)
            share = positives / len(examples) if examples else 0.0
            print(
                f"эпизод {episode + 1}/{episodes}: {len(examples)} примеров, "
                f"need_deep={share:.2f}",
                flush=True,
            )
    return examples


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/maze/maze.jsonl"))
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--epsilon", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    examples = generate(
        episodes=args.episodes,
        steps=args.steps,
        horizon=args.horizon,
        stride=args.stride,
        epsilon=args.epsilon,
        seed=args.seed,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example, ensure_ascii=False) + "\n")
    positives = sum(ex["labels"]["need_deep"] for ex in examples)
    print(f"-> {args.out}: {len(examples)} примеров, need_deep={positives}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
