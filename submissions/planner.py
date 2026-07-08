from __future__ import annotations

import heapq
from typing import Iterable

import numpy as np

from nesylink.core.constants import (
    ACTION_DOWN,
    ACTION_LEFT,
    ACTION_RIGHT,
    ACTION_UP,
    GRID_HEIGHT,
    GRID_WIDTH,
)


GridPos = tuple[int, int]

_MOVE_DELTAS: dict[GridPos, int] = {
    (0, -1): ACTION_UP,
    (0, 1): ACTION_DOWN,
    (-1, 0): ACTION_LEFT,
    (1, 0): ACTION_RIGHT,
}


def build_occupancy(symbol_map) -> np.ndarray:
    

    occupancy = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=np.uint8)
    blocked = set(symbol_map.blocked_tiles()) | set(symbol_map.monsters)
    for x, y in blocked:
        if 0 <= x < GRID_WIDTH and 0 <= y < GRID_HEIGHT:
            occupancy[y, x] = 1
    return occupancy


def a_star(grid: np.ndarray, start: GridPos, goal: GridPos) -> list[GridPos]:
    
    occupancy = np.asarray(grid)
    if occupancy.shape != (GRID_HEIGHT, GRID_WIDTH):
        raise ValueError(f"grid shape must be {(GRID_HEIGHT, GRID_WIDTH)}, got {occupancy.shape}")
    if not _in_bounds(start) or not _in_bounds(goal):
        return []
    if start == goal:
        return []
    if occupancy[start[1], start[0]] != 0 or occupancy[goal[1], goal[0]] != 0:
        return []

    frontier: list[tuple[int, int, GridPos]] = []
    heapq.heappush(frontier, (_manhattan(start, goal), 0, start))
    came_from: dict[GridPos, GridPos | None] = {start: None}
    cost_so_far: dict[GridPos, int] = {start: 0}

    while frontier:
        _, current_cost, current = heapq.heappop(frontier)
        if current == goal:
            return _reconstruct_path(came_from, goal)
        if current_cost != cost_so_far[current]:
            continue

        for nxt in _neighbors(current):
            if occupancy[nxt[1], nxt[0]] != 0:
                continue
            new_cost = current_cost + 1
            if nxt not in cost_so_far or new_cost < cost_so_far[nxt]:
                cost_so_far[nxt] = new_cost
                came_from[nxt] = current
                priority = new_cost + _manhattan(nxt, goal)
                heapq.heappush(frontier, (priority, new_cost, nxt))

    return []


def path_to_actions(path: list[GridPos]) -> list[int]:
    

    actions: list[int] = []
    if len(path) < 2:
        return actions
    for current, nxt in zip(path, path[1:]):
        delta = (nxt[0] - current[0], nxt[1] - current[1])
        if delta not in _MOVE_DELTAS:
            raise ValueError(f"path contains non-adjacent steps: {current} -> {nxt}")
        actions.append(_MOVE_DELTAS[delta])
    return actions


def plan_path(symbol_map, start: GridPos, goal: GridPos) -> list[int]:
    

    if start == goal:
        return []
    if not _in_bounds(start) or not _in_bounds(goal):
        return []

    occupancy = build_occupancy(symbol_map)
    occupancy[start[1], start[0]] = 0

    blocked = set(symbol_map.blocked_tiles()) | set(symbol_map.monsters)
    if goal not in blocked:
        occupancy[goal[1], goal[0]] = 0
        path = a_star(occupancy, start, goal)
        return path_to_actions([start, *path]) if path else []

    best_path: list[GridPos] = []
    for adjacent in sorted(_neighbors(goal), key=lambda pos: _manhattan(start, pos)):
        if occupancy[adjacent[1], adjacent[0]] != 0:
            continue
        path = a_star(occupancy, start, adjacent)
        if path and (not best_path or len(path) < len(best_path)):
            best_path = path
    return path_to_actions([start, *best_path]) if best_path else []


def nearest_reachable_goal(symbol_map, start: GridPos, goals: Iterable[GridPos]) -> GridPos | None:
    

    best_goal: GridPos | None = None
    best_len: int | None = None
    for goal in goals:
        actions = plan_path(symbol_map, start, goal)
        if goal == start:
            return goal
        if actions and (best_len is None or len(actions) < best_len):
            best_goal = goal
            best_len = len(actions)
    return best_goal


def _neighbors(pos: GridPos) -> Iterable[GridPos]:
    x, y = pos
    for dx, dy in _MOVE_DELTAS:
        nxt = (x + dx, y + dy)
        if _in_bounds(nxt):
            yield nxt


def _in_bounds(pos: GridPos) -> bool:
    x, y = pos
    return 0 <= x < GRID_WIDTH and 0 <= y < GRID_HEIGHT


def _manhattan(left: GridPos, right: GridPos) -> int:
    return abs(left[0] - right[0]) + abs(left[1] - right[1])


def _reconstruct_path(came_from: dict[GridPos, GridPos | None], goal: GridPos) -> list[GridPos]:
    path: list[GridPos] = []
    cursor: GridPos | None = goal
    while cursor is not None:
        path.append(cursor)
        cursor = came_from[cursor]
    path.reverse()
    return path[1:]
