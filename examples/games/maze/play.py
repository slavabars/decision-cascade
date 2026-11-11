"""Партия в лабиринте: голова решает, когда жать на осторожный маршрут.

Голова `need_deep` отвечает на вопрос «жадный шаг убьёт в ближайшие 12 ходов?».
Пока вероятность ниже порога — идём кратчайшим путём к точке и не платим за
просчёт. Выше — считаем осторожный маршрут в обход призраков.

    python -m examples.games.maze.play --pack checkpoints/packs/maze \
        --seed 3 --demo

Демо-режим печатает кадры с паузами и цветом — из него собирается GIF.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from api.server.service import DecisionService, HashStateEncoder
from examples.games.maze.env import Maze
from examples.games.maze.planner import cautious_move, greedy_move
from examples.games.maze.specs import MAZE_SPECS
from examples.games.maze.state import render, text_state

CLEAR = "\x1b[2J\x1b[H"
DIM = "\x1b[90m"
BOLD = "\x1b[1m"
GREEN = "\x1b[32m"
YELLOW = "\x1b[33m"
RESET = "\x1b[0m"


def load_service(pack: Path) -> DecisionService:
    metadata = pack / "metadata.json"
    if not metadata.is_file():
        raise SystemExit(
            f"нет пака голов в {pack}: сначала обучите его\n"
            "  python -m examples.games.maze.dataset\n"
            f"  python -m examples.games.maze.train --data data/maze/maze.jsonl "
            f"--out {pack}"
        )
    dim = json.loads(metadata.read_text(encoding="utf-8"))["feature_dim"]
    return DecisionService(
        encoder=HashStateEncoder(dim),
        dim=dim,
        checkpoint_dir=pack,
        untrained_policy="refuse",
    )


def play(
    *,
    pack: Path,
    seed: int = 0,
    max_steps: int = 300,
    threshold: float = 0.4,
    demo: bool = False,
    delay: float = 0.08,
) -> dict:
    service = load_service(pack)
    game = Maze(seed=seed, max_steps=max_steps)
    escalations = moves = 0
    started = time.perf_counter()
    while not game.done:
        state = text_state(game)
        probability = float(service.decide(state, MAZE_SPECS)["need_deep"].noul)
        escalated = probability >= threshold
        if escalated:
            move, _ = cautious_move(game)
            escalations += 1
        else:
            move, _ = greedy_move(game)
        moves += 1
        game.step(move)

        if demo:
            print(CLEAR, end="")
            print(f"{BOLD}decision-cascade · лабиринт{RESET}")
            print(render(game))
            verdict = (
                f"{YELLOW}need_deep={probability:.2f} → осторожный маршрут{RESET}"
                if escalated
                else f"{GREEN}need_deep={probability:.2f} → жадный шаг{RESET}"
            )
            print(verdict)
            print(
                f"{DIM}очки {game.score} · жизни {game.lives} · "
                f"шаг {game.steps}/{max_steps} · "
                f"эскалаций {escalations}/{moves}{RESET}"
            )
            time.sleep(delay)

    elapsed = time.perf_counter() - started
    return {
        "score": game.score,
        "deaths": game.deaths,
        "steps": game.steps,
        "escalation": escalations / max(1, moves),
        "seconds": elapsed,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, default=Path("checkpoints/packs/maze"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=300)
    # порог выбран по кривой из evaluate.py (см. README примера)
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--demo", action="store_true", help="цвет, паузы, очистка экрана")
    parser.add_argument("--delay", type=float, default=0.08)
    args = parser.parse_args(argv)

    stats = play(
        pack=args.pack,
        seed=args.seed,
        max_steps=args.max_steps,
        threshold=args.threshold,
        demo=args.demo,
        delay=args.delay,
    )
    print(
        f"\n{BOLD}Итог: {stats['score']} очков · {stats['deaths']} смертей · "
        f"{stats['escalation']:.0%} эскалаций · {stats['steps']} шагов{RESET}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
