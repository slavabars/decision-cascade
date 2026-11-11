"""Две политики: жадный шаг и осторожный маршрут.

Жадная политика идёт кратчайшим путём к ближайшей точке и не смотрит на
призраков вообще — она дешёвая и именно поэтому регулярно погибает. Осторожная
обходит клетки рядом с призраками и стоит дороже: это и есть «глубокий просчёт»,
который каскад включает по решению головы.
"""
from __future__ import annotations

import heapq
from collections import deque

from examples.games.maze.env import DIRECTIONS, Maze, Pos

#: Клетки на таком расстоянии от призрака осторожная политика считает опасными.
DANGER_RADIUS = 1


def _targets(maze: Maze) -> set[Pos]:
    """Что собирать: точки, энергайзеры и бонус, пока он на карте."""
    targets = set(maze.pellets) | set(maze.powers)
    if maze.bonus is not None:
        targets.add(maze.bonus)
    return targets


def _first_step_to(
    maze: Maze, targets: set[Pos], *, blocked: set[Pos] | None = None
) -> tuple[Pos | None, int]:
    """Первый шаг BFS к ближайшей цели. Возвращает (направление, узлов просмотрено)."""
    blocked = blocked or set()
    if maze.hero in targets:
        return None, 0
    seen = {maze.hero}
    queue: deque[tuple[Pos, Pos | None]] = deque([(maze.hero, None)])
    nodes = 0
    while queue:
        pos, first = queue.popleft()
        nodes += 1
        if pos in targets and first is not None:
            return first, nodes
        for direction in DIRECTIONS:
            nxt = (pos[0] + direction[0], pos[1] + direction[1])
            if nxt in seen or maze.is_wall(nxt) or nxt in blocked:
                continue
            seen.add(nxt)
            queue.append((nxt, first or direction))
    return None, nodes


def _danger_cells(maze: Maze) -> set[Pos]:
    """Клетки вокруг призраков: там осторожная политика не ходит без нужды."""
    cells: set[Pos] = set()
    for ghost in maze.ghosts:
        for dx in range(-DANGER_RADIUS, DANGER_RADIUS + 1):
            for dy in range(-DANGER_RADIUS, DANGER_RADIUS + 1):
                if abs(dx) + abs(dy) <= DANGER_RADIUS:
                    cells.add((ghost.pos[0] + dx, ghost.pos[1] + dy))
    return cells


def _fallback(maze: Maze) -> Pos:
    """Если пути нет — уходим от ближайшего призрака."""
    moves = maze.legal_moves(maze.hero)
    if not moves:
        return DIRECTIONS[0]
    ghosts = maze.ghost_positions()
    return max(
        moves,
        key=lambda direction: min(
            abs(maze.hero[0] + direction[0] - ghost[0])
            + abs(maze.hero[1] + direction[1] - ghost[1])
            for ghost in ghosts
        ),
    )


def greedy_move(maze: Maze) -> tuple[Pos, int]:
    """Кратчайший путь к ближайшей точке, призраки игнорируются."""
    direction, nodes = _first_step_to(maze, _targets(maze))
    return (direction or _fallback(maze)), nodes


def cautious_move(maze: Maze, *, danger_cost: int = 4) -> tuple[Pos, int]:
    """Взвешенный путь: близость призрака дорогая, но не запрещённая.

    Жёстко блокировать окрестность призраков нельзя: в узком коридоре это
    отрезает героя от точек, и он начинает кружить. Поэтому клетка рядом с
    призраком стоит `danger_cost` вместо 1, а клетка самого призрака недоступна.
    """
    targets = _targets(maze)
    danger = _danger_cells(maze) - set(maze.ghost_positions())
    blocked = set(maze.ghost_positions())
    best: dict[Pos, int] = {maze.hero: 0}
    first: dict[Pos, Pos | None] = {maze.hero: None}
    heap: list[tuple[int, int, int, Pos]] = [(0, 0, 0, maze.hero)]
    nodes = 0
    counter = 0
    while heap:
        cost, _, _, pos = heapq.heappop(heap)
        nodes += 1
        if cost > best.get(pos, 1 << 30):
            continue
        if pos in targets and first[pos] is not None:
            return first[pos], nodes
        for direction in DIRECTIONS:
            nxt = (pos[0] + direction[0], pos[1] + direction[1])
            if maze.is_wall(nxt) or nxt in blocked:
                continue
            step = danger_cost if nxt in danger else 1
            candidate = cost + step
            if candidate < best.get(nxt, 1 << 30):
                best[nxt] = candidate
                first[nxt] = first[pos] or direction
                counter += 1
                heapq.heappush(heap, (candidate, counter, 0, nxt))
    return _fallback(maze), nodes
