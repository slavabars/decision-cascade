"""Оценка: сколько очков стоит осторожность и сколько эскалаций она требует."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from api.server.service import DecisionService, HashStateEncoder
from examples.games.maze.env import Maze
from examples.games.maze.planner import cautious_move, greedy_move
from examples.games.maze.specs import MAZE_SPECS
from examples.games.maze.state import text_state


def load_service(pack: Path) -> DecisionService:
    metadata = json.loads((pack / "metadata.json").read_text(encoding="utf-8"))
    dim = metadata["feature_dim"]
    return DecisionService(
        encoder=HashStateEncoder(dim),
        dim=dim,
        checkpoint_dir=pack,
        untrained_policy="refuse",
    )


def decide(service: DecisionService, game: Maze) -> float:
    answer = service.decide(text_state(game), MAZE_SPECS)["need_deep"]
    return float(answer.noul)


def run_episode(
    policy: str,
    *,
    seed: int,
    max_steps: int,
    service: DecisionService | None = None,
    threshold: float = 0.5,
) -> dict[str, float]:
    game = Maze(seed=seed, max_steps=max_steps)
    moves = cautious_calls = nodes = 0
    while not game.done:
        if policy == "greedy":
            move, cost = greedy_move(game)
        elif policy == "cautious":
            move, cost = cautious_move(game)
            cautious_calls += 1
        else:
            if service is None:
                raise ValueError("cascade needs a trained pack")
            probability = decide(service, game)
            if probability >= threshold:
                move, cost = cautious_move(game)
                cautious_calls += 1
            else:
                move, cost = greedy_move(game)
        nodes += cost
        moves += 1
        game.step(move)
    return {
        "score": float(game.score),
        "deaths": float(game.deaths),
        "escalation": cautious_calls / max(1, moves),
        "nodes": nodes / max(1, moves),
        "steps": float(game.steps),
    }


def summarize(rows: list[dict[str, float]]) -> dict[str, float]:
    return {
        key: statistics.mean(row[key] for row in rows)
        for key in ("score", "deaths", "escalation", "nodes")
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, default=Path("checkpoints/packs/maze"))
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--thresholds", default="0.2,0.3,0.5")
    args = parser.parse_args(argv)

    service = load_service(args.pack)
    seeds = range(args.episodes)
    print("| политика | очки | смерти | эскалаций | узлов/ход |")
    print("| --- | ---: | ---: | ---: | ---: |")

    for name, policy, threshold in (
        ("жадная", "greedy", 0.0),
        ("осторожная", "cautious", 1.0),
    ):
        rows = [
            run_episode(policy, seed=seed, max_steps=args.max_steps)
            for seed in seeds
        ]
        stats = summarize(rows)
        print(
            f"| {name} | {stats['score']:.0f} | {stats['deaths']:.2f} | "
            f"{stats['escalation']:.0%} | {stats['nodes']:.0f} |"
        )

    for threshold in (float(x) for x in args.thresholds.split(",") if x.strip()):
        rows = [
            run_episode(
                "cascade",
                seed=seed,
                max_steps=args.max_steps,
                service=service,
                threshold=threshold,
            )
            for seed in seeds
        ]
        stats = summarize(rows)
        print(
            f"| каскад {threshold:.1f} | {stats['score']:.0f} | {stats['deaths']:.2f} | "
            f"{stats['escalation']:.0%} | {stats['nodes']:.0f} |"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
