"""Детерминированный лабиринт: сетка, точки, призраки, три жизни.

Своя среда, а не ncurses-клон: у консольных пакманов нет машинного интерфейса,
поэтому «играть» в них можно только скрейпингом экрана. Здесь состояние — это
данные, эпизод воспроизводится по seed, а решение проверяется очками и смертями.

Мир намеренно простой: одна карта, три призрака, погоня по манхэттену с
детерминированным выбором на развилках. Этого достаточно, чтобы решение было
нетривиальным (жадный шаг к точке регулярно ведёт в призрака), но не требует
симулятора уровня аркады.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

#: Карта: `#` — стена, `.` — точка, `o` — энергайзер, ` ` — пусто,
#: `P` — старт героя, `G` — старт призрака.
MAZE = (
    "###################",
    "#........#........#",
    "#o##.###.#.###.##o#",
    "#.................#",
    "#.##.#.#####.#.##.#",
    "#....#...#...#....#",
    "####.###.#.###.####",
    "#...G..G.#.G......#",
    "####.###.#.###.####",
    "#....#...#...#....#",
    "#.##.#.#####.#.##.#",
    "#........P........#",
    "#o##.###.#.###.##o#",
    "#........#........#",
    "###################",
)

WALL = "#"
PELLET = "."
POWER = "o"

UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)
DIRECTIONS = (UP, DOWN, LEFT, RIGHT)

PELLET_SCORE = 10
POWER_SCORE = 50
GHOST_SCORE = 200
BONUS_SCORE = 100
FRIGHT_TICKS = 30

#: Бонус (как фрукт в аркаде) появляется раз в `BONUS_EVERY` шагов на `BONUS_SPOT`
#: и живёт `BONUS_LIFETIME` шагов, после чего исчезает, если его не съели.
BONUS_EVERY = 70
BONUS_LIFETIME = 45
BONUS_SPOT: Pos = (9, 3)

Pos = tuple[int, int]


def load_maze(rows: tuple[str, ...] = MAZE) -> tuple[frozenset[Pos], set[Pos], set[Pos], Pos, list[Pos]]:
    """Разбирает карту: (стены, точки, энергайзеры, старт героя, старты призраков)."""
    walls: set[Pos] = set()
    pellets: set[Pos] = set()
    powers: set[Pos] = set()
    hero: Pos | None = None
    ghosts: list[Pos] = []
    for y, row in enumerate(rows):
        for x, char in enumerate(row):
            pos = (x, y)
            if char == WALL:
                walls.add(pos)
            elif char == PELLET:
                pellets.add(pos)
            elif char == POWER:
                powers.add(pos)
            elif char == "P":
                hero = pos
            elif char == "G":
                ghosts.append(pos)
    if hero is None:
        raise ValueError("в карте нет старта героя (P)")
    return frozenset(walls), pellets, powers, hero, ghosts


@dataclass
class Ghost:
    pos: Pos
    direction: Pos = RIGHT
    scared: int = 0
    home: Pos = field(default=RIGHT)

    def copy(self) -> Ghost:
        return Ghost(self.pos, self.direction, self.scared, self.home)


@dataclass
class StepResult:
    reward: int
    done: bool
    ate_ghost: bool = False
    died: bool = False


class Maze:
    """Эпизод лабиринта: `step(direction)` двигает героя и призраков."""

    def __init__(
        self,
        seed: int = 0,
        *,
        max_steps: int = 400,
        lives: int = 3,
        ghost_speed: int = 1,
        ghost_scatter: float = 0.3,
    ):
        self.rows = MAZE
        self.ghost_speed = ghost_speed
        #: Доля случайных ходов у призрака: без неё погоня не оставляет шансов.
        self.ghost_scatter = ghost_scatter
        (
            self.walls,
            self.pellets,
            self.powers,
            self.hero_start,
            _ghost_starts,
        ) = load_maze()
        self.seed = seed
        self.rng = random.Random(seed)
        self.max_steps = max_steps
        self.lives = lives
        self.reset()

    # --- состояние ----------------------------------------------------------
    def reset(self) -> None:
        _, self.pellets, self.powers, self.hero_start, ghost_starts = load_maze()
        self.hero = self.hero_start
        self.ghosts = [Ghost(pos, home=pos) for pos in ghost_starts]
        self.score = 0
        self.steps = 0
        self.lives = 3
        self.deaths = 0
        self.fright_ticks = 0
        self.bonus: Pos | None = None
        self.bonus_ticks = 0
        self.done = False
        self.rng = random.Random(self.seed)

    @property
    def pellets_left(self) -> int:
        return len(self.pellets) + len(self.powers)

    def is_wall(self, pos: Pos) -> bool:
        return pos in self.walls

    def legal_moves(self, pos: Pos) -> list[Pos]:
        moves = []
        for direction in DIRECTIONS:
            nxt = (pos[0] + direction[0], pos[1] + direction[1])
            if not self.is_wall(nxt):
                moves.append(direction)
        return moves

    def maze_rows(self) -> tuple[str, ...]:
        """Исходная карта — для рендера (точки и энергайзеры живут отдельно)."""
        return self.rows

    def ghost_positions(self) -> list[Pos]:
        return [ghost.pos for ghost in self.ghosts]

    # --- шаг ----------------------------------------------------------------
    def step(self, direction: Pos) -> StepResult:
        if self.done:
            return StepResult(0, True)
        self.steps += 1
        reward = 0

        target = (self.hero[0] + direction[0], self.hero[1] + direction[1])
        if not self.is_wall(target):
            self.hero = target
        if self.hero in self.pellets:
            self.pellets.discard(self.hero)
            reward += PELLET_SCORE
        if self.hero in self.powers:
            self.powers.discard(self.hero)
            reward += POWER_SCORE
            self.fright_ticks = FRIGHT_TICKS
            for ghost in self.ghosts:
                ghost.scared = FRIGHT_TICKS
        if self.bonus is not None and self.hero == self.bonus:
            reward += BONUS_SCORE
            self.bonus = None
            self.bonus_ticks = 0

        for ghost in self.ghosts:
            for _ in range(self.ghost_speed):
                self._move_ghost(ghost)

        self._tick_bonus()

        if self.fright_ticks > 0:
            self.fright_ticks -= 1
            for ghost in self.ghosts:
                ghost.scared = max(0, ghost.scared - 1)

        ate_ghost = False
        for ghost in self.ghosts:
            if ghost.pos != self.hero:
                continue
            if ghost.scared > 0:
                ghost.pos = ghost.home
                ghost.scared = 0
                reward += GHOST_SCORE
                ate_ghost = True
            else:
                self._die()
                self.score += reward
                return StepResult(reward, self.done, ate_ghost, died=True)

        if self.pellets_left == 0:
            self.done = True
        if self.steps >= self.max_steps:
            self.done = True
        self.score += reward
        return StepResult(reward, self.done, ate_ghost)

    def _tick_bonus(self) -> None:
        if self.bonus is not None:
            self.bonus_ticks -= 1
            if self.bonus_ticks <= 0:
                self.bonus = None
        if self.bonus is None and self.steps % BONUS_EVERY == 0:
            self.bonus = BONUS_SPOT
            self.bonus_ticks = BONUS_LIFETIME

    def _die(self) -> None:
        self.deaths += 1
        self.lives -= 1
        if self.lives <= 0:
            self.done = True
            return
        self.hero = self.hero_start
        for ghost in self.ghosts:
            ghost.pos = ghost.home
            ghost.scared = 0
        self.fright_ticks = 0

    def _move_ghost(self, ghost: Ghost) -> None:
        options = self.legal_moves(ghost.pos)
        if not options:
            return
        # Призрак не разворачивается на 180°, но и не обязан сворачивать:
        # иначе в прямом коридоре он дёргается на месте вместо погони.
        reverse = (-ghost.direction[0], -ghost.direction[1])
        if len(options) > 1:
            options = [move for move in options if move != reverse] or options
        if ghost.scared > 0 or self.rng.random() < self.ghost_scatter:
            choice = self.rng.choice(options)
        else:
            target = self.hero
            distances = [
                abs(ghost.pos[0] + move[0] - target[0])
                + abs(ghost.pos[1] + move[1] - target[1])
                for move in options
            ]
            best = min(distances)
            choice = self.rng.choice(
                [move for move, distance in zip(options, distances) if distance == best]
            )
        ghost.direction = choice
        ghost.pos = (ghost.pos[0] + choice[0], ghost.pos[1] + choice[1])

    # --- удобства -----------------------------------------------------------
    def copy(self) -> Maze:
        clone = Maze.__new__(Maze)
        clone.walls = self.walls
        clone.pellets = set(self.pellets)
        clone.powers = set(self.powers)
        clone.hero_start = self.hero_start
        clone.seed = self.seed
        clone.rng = random.Random()
        clone.rng.setstate(self.rng.getstate())
        clone.max_steps = self.max_steps
        clone.ghost_speed = self.ghost_speed
        clone.ghost_scatter = self.ghost_scatter
        clone.hero = self.hero
        clone.ghosts = [ghost.copy() for ghost in self.ghosts]
        clone.score = self.score
        clone.steps = self.steps
        clone.lives = self.lives
        clone.deaths = self.deaths
        clone.fright_ticks = self.fright_ticks
        clone.bonus = self.bonus
        clone.bonus_ticks = self.bonus_ticks
        clone.done = self.done
        return clone
