"""Признаки для головы и рендер лабиринта.

Признаки — только то, что видно герою до хода: окно вокруг него, расстояния до
призраков и до ближайшей точки, число выходов из клетки. Ничего из будущего
(сколько шагов проживёт герой) в состояние не попадает — иначе метка протекла бы
в признаки.
"""
from __future__ import annotations

from collections import deque

from examples.games.maze.env import (
    DIRECTIONS,
    PELLET,
    POWER,
    WALL,
    Maze,
    Pos,
)

EMPTY = " "
HERO = "H"
GHOST = "G"

#: Ширина окна вокруг героя (нечётная, чтобы герой был в центре).
WINDOW = 5


def _bucket(value: int, edges: tuple[int, ...], labels: tuple[str, ...]) -> str:
    for edge, label in zip(edges, labels):
        if value <= edge:
            return label
    return labels[-1]


def _manhattan(a: Pos, b: Pos) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _window(maze: Maze) -> list[str]:
    """Окно WINDOW×WINDOW вокруг героя, по строке на токен."""
    half = WINDOW // 2
    ghosts = set(maze.ghost_positions())
    rows = []
    for dy in range(-half, half + 1):
        cells = []
        for dx in range(-half, half + 1):
            pos = (maze.hero[0] + dx, maze.hero[1] + dy)
            if dx == 0 and dy == 0:
                cells.append(HERO)
            elif pos in ghosts:
                cells.append(GHOST)
            elif pos in maze.walls:
                cells.append(WALL)
            elif pos in maze.pellets:
                cells.append(PELLET)
            elif pos in maze.powers:
                cells.append(POWER)
            else:
                cells.append(EMPTY)
        rows.append("".join(cells).replace(" ", "_"))
    return rows


def _distance_to_nearest_pellet(maze: Maze) -> int:
    """BFS по проходам до ближайшей точки; 99 — если недостижимых нет."""
    targets = maze.pellets | maze.powers
    if not targets:
        return 0
    seen = {maze.hero}
    queue = deque([(maze.hero, 0)])
    while queue:
        pos, distance = queue.popleft()
        if pos in targets:
            return distance
        for dx, dy in DIRECTIONS:
            nxt = (pos[0] + dx, pos[1] + dy)
            if nxt not in seen and not maze.is_wall(nxt):
                seen.add(nxt)
                queue.append((nxt, distance + 1))
    return 99


def text_state(maze: Maze) -> str:
    """Компактное состояние: окно, призраки, выходы, фаза — токенами."""
    ghosts = sorted(maze.ghost_positions(), key=lambda pos: _manhattan(maze.hero, pos))
    distances = [_manhattan(maze.hero, pos) for pos in ghosts]
    near = sum(1 for distance in distances if distance <= 3)
    tokens = [f"maze win{i}={row}" for i, row in enumerate(_window(maze))]
    tokens += [
        f"ghosts={'none' if not ghosts else ','.join(str(d) for d in distances[:3])}",
        f"ghosts_near={_bucket(near, (0, 1, 2), ('0', '1', '2', '3+'))}",
        f"hero_ghost={_bucket(distances[0], (1, 2, 3, 5), ('1', '2', '3', '4-5', '6+'))}",
        f"pellet_dist={_bucket(_distance_to_nearest_pellet(maze), (1, 3, 6, 10), ('1', '2-3', '4-6', '7-10', '11+'))}",
        f"exits={len(maze.legal_moves(maze.hero))}",
        f"power={'yes' if maze.fright_ticks > 0 else 'no'}",
        f"bonus={'yes' if maze.bonus is not None else 'no'}",
        f"left={_bucket(maze.pellets_left, (10, 40, 80, 120), ('0-10', '11-40', '41-80', '81-120', '121+'))}",
    ]
    return " ".join(tokens)


_RESET = "\x1b[0m"
_BLUE = "\x1b[94m"
_CYAN = "\x1b[96m"
_DIM = "\x1b[90m"
_YELLOW = "\x1b[93m"
_WHITE = "\x1b[97m"
_RED = "\x1b[91m"
_MAGENTA = "\x1b[95m"


def render(maze: Maze, *, color: bool = True) -> str:
    """Лабиринт двумя символами на клетку: читаемо и одинаково во всех шрифтах."""
    ghosts = set(maze.ghost_positions())
    lines = []
    for y, row in enumerate(maze.maze_rows()):
        cells = []
        for x, char in enumerate(row):
            pos = (x, y)
            if pos == maze.hero:
                cell, tint = "██", _YELLOW
            elif pos in ghosts:
                scared = any(ghost.scared > 0 for ghost in maze.ghosts if ghost.pos == pos)
                cell, tint = "██", (_MAGENTA if scared else _RED)
            elif char == WALL:
                cell, tint = "██", _BLUE
            elif maze.bonus is not None and pos == maze.bonus:
                cell, tint = "$ ", _CYAN
            elif pos in maze.powers:
                cell, tint = "● ", _WHITE
            elif pos in maze.pellets:
                cell, tint = "· ", _DIM
            else:
                # съеденные точки и энергайзеры на карте не рисуются
                cell, tint = "  ", ""
            cells.append(f"{tint}{cell}{_RESET}" if color and tint else cell)
        lines.append("".join(cells))
    return "\n".join(lines)
