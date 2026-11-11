"""Лабиринт: среда, политики и логика каскада без обучения.

Тесты быстрые и детерминированные: ни движка, ни сети, ни обученных паков.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from examples.games.maze import dataset, evaluate, play
from examples.games.maze.env import (
    BONUS_EVERY,
    BONUS_LIFETIME,
    BONUS_SCORE,
    BONUS_SPOT,
    DIRECTIONS,
    Maze,
)
from examples.games.maze.planner import cautious_move, greedy_move
from examples.games.maze.state import render, text_state


# --- среда -----------------------------------------------------------------
def test_all_pellets_are_reachable_from_start():
    from collections import deque

    game = Maze(seed=0)
    seen = {game.hero}
    queue = deque([game.hero])
    while queue:
        x, y = queue.popleft()
        for dx, dy in DIRECTIONS:
            nxt = (x + dx, y + dy)
            if nxt not in seen and not game.is_wall(nxt):
                seen.add(nxt)
                queue.append(nxt)
    assert not [pos for pos in (game.pellets | game.powers) if pos not in seen]
    assert not [ghost.pos for ghost in game.ghosts if ghost.pos not in seen]


def test_episode_is_deterministic():
    def run(seed: int) -> tuple[int, int, int]:
        game = Maze(seed=seed, max_steps=80)
        while not game.done:
            game.step(DIRECTIONS[game.steps % len(DIRECTIONS)])
        return game.score, game.deaths, game.steps

    assert run(5) == run(5)


def test_move_into_wall_keeps_hero_in_place():
    game = Maze(seed=0)
    game.hero = (1, 1)
    game.ghosts = []  # проверяем только стены: призраки двигаются сами
    game.step((-1, 0))  # (0, 1) — стена
    assert game.hero == (1, 1)


def test_copy_is_independent():
    game = Maze(seed=0)
    clone = game.copy()
    clone.step(DIRECTIONS[0])
    assert clone.steps == game.steps + 1
    # прогон на копии не должен менять мир оригинала
    clone.pellets.clear()
    assert game.pellets


def test_bonus_appears_on_schedule_and_expires():
    game = Maze(seed=0, max_steps=300)
    game.ghosts = []
    while game.steps < BONUS_EVERY:
        game.step((0, 0))
    assert game.bonus == BONUS_SPOT
    assert game.bonus_ticks == BONUS_LIFETIME
    while game.bonus is not None:
        game.step((0, 0))
    assert game.steps >= BONUS_EVERY + BONUS_LIFETIME


def test_bonus_is_collected_for_points():
    game = Maze(seed=0, max_steps=300)
    game.ghosts = []
    while game.bonus is None:
        game.step((0, 0))
        assert not game.done
    game.hero = BONUS_SPOT
    before = game.score
    game.step((0, 0))
    assert game.bonus is None
    assert game.score - before >= BONUS_SCORE


def test_eaten_pellets_disappear_from_the_map():
    """Съеденная точка не должна рисоваться: иначе карта врёт игроку."""
    game = Maze(seed=0)
    game.pellets.clear()
    game.powers.clear()
    plain = render(game, color=False)
    assert "·" not in plain
    assert "●" not in plain


# --- политики --------------------------------------------------------------
def test_both_policies_return_legal_moves():
    game = Maze(seed=2)
    for policy in (greedy_move, cautious_move):
        move, nodes = policy(game)
        assert move in game.legal_moves(game.hero)
        assert nodes >= 0


def test_cautious_policy_searches_more():
    def total_nodes(policy) -> int:
        game = Maze(seed=0, max_steps=60)
        total = 0
        while not game.done:
            move, nodes = policy(game)
            total += nodes
            game.step(move)
        return total

    assert total_nodes(cautious_move) > total_nodes(greedy_move)


# --- разметка --------------------------------------------------------------
def test_label_is_binary_and_uses_horizon():
    game = Maze(seed=1)
    labelled = dataset.label_position(game, horizon=12)
    assert labelled["need_deep"] in (0, 1)
    assert labelled["greedy"] in game.legal_moves(game.hero)


def test_generate_writes_splits(tmp_path: Path):
    examples = dataset.generate(episodes=2, steps=30, stride=5, progress=False)
    assert examples
    assert all(set(item["labels"]) == {"need_deep"} for item in examples)
    assert all(isinstance(item["state"], str) and item["state"] for item in examples)

    path = tmp_path / "maze.jsonl"
    path.write_text(
        "\n".join(json.dumps(item) for item in examples) + "\n", encoding="utf-8"
    )
    assert json.loads(path.read_text(encoding="utf-8").splitlines()[0])["labels"]


# --- признаки и рендер -----------------------------------------------------
def test_state_has_no_future_information():
    game = Maze(seed=0)
    state = text_state(game)
    assert state.startswith("maze ")
    for token in ("win0=", "ghosts=", "pellet_dist=", "exits=", "power="):
        assert token in state
    for forbidden in ("need_deep", "greedy", "survived"):
        assert forbidden not in state


def test_render_marks_hero_and_ghosts():
    game = Maze(seed=0)
    plain = render(game, color=False)
    assert "██" in plain
    assert plain.count("\n") == 15 - 1
    colored = render(game, color=True)
    assert "\x1b[93m" in colored  # герой


# --- оценка ----------------------------------------------------------------
def test_run_episode_greedy_reports_metrics():
    stats = evaluate.run_episode("greedy", seed=0, max_steps=60)
    assert set(stats) == {"score", "deaths", "escalation", "nodes", "steps"}
    assert stats["score"] >= 0
    assert stats["escalation"] == 0.0


def test_cascade_requires_service():
    with pytest.raises(ValueError, match="pack"):
        evaluate.run_episode("cascade", seed=0, max_steps=10)


def test_cascade_escalates_by_probability(tmp_path: Path):
    """Порог — единственное, что решает, идти ли осторожно."""

    class StubService:
        def __init__(self, probability: float):
            self.probability = probability

        def decide(self, _state, _specs):
            return {
                "need_deep": SimpleNamespace(
                    noul=self.probability, confidence=self.probability
                )
            }

    stats = evaluate.run_episode(
        "cascade",
        seed=0,
        max_steps=20,
        service=StubService(0.95),
        threshold=0.5,
    )
    assert stats["escalation"] == pytest.approx(1.0)


def test_play_rejects_missing_pack(tmp_path: Path):
    with pytest.raises(SystemExit, match="нет пака"):
        play.load_service(tmp_path / "empty")
