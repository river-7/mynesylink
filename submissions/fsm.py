from __future__ import annotations

from enum import Enum
from typing import Optional


GridPos = tuple[int, int]


class TaskState(Enum):
    GET_KEY = 1
    OPEN_CHEST = 2
    KILL_GUARDIAN = 3
    GO_TO_EXIT = 4
    DONE = 5
    PRESS_BUTTON = 6


def get_initial_state(task_id: str) -> TaskState:
    
    del task_id
    return TaskState.GET_KEY


def get_goal(state: TaskState, symbol_map) -> Optional[GridPos]:
    
    player = symbol_map.player
    if state in {TaskState.GET_KEY, TaskState.OPEN_CHEST}:
        return _nearest(player, symbol_map.chests)
    if state == TaskState.KILL_GUARDIAN:
        return _nearest(player, symbol_map.monsters)
    if state == TaskState.PRESS_BUTTON:
        return _nearest(player, tuple(symbol_map.buttons) + tuple(symbol_map.switches))
    if state == TaskState.GO_TO_EXIT:
        return _nearest(player, symbol_map.exits)
    return None


def update_state(state: TaskState, symbol_map, inventory: dict) -> TaskState:
   
    if state == TaskState.DONE:
        return TaskState.DONE

    key_count = _inventory_int(inventory, "key", "keys")
    has_sword = _inventory_bool(inventory, "sword", "has_sword")

    if state == TaskState.GET_KEY:
        if key_count > 0:
            if symbol_map.monsters and has_sword:
                return TaskState.KILL_GUARDIAN
            if symbol_map.monsters and symbol_map.chests and not has_sword:
                return TaskState.OPEN_CHEST
            return TaskState.GO_TO_EXIT
        if symbol_map.chests:
            return TaskState.GET_KEY
        return TaskState.GO_TO_EXIT if symbol_map.exits else TaskState.GET_KEY

    if state == TaskState.OPEN_CHEST:
        if has_sword:
            return TaskState.KILL_GUARDIAN if symbol_map.monsters else TaskState.GO_TO_EXIT
        if symbol_map.chests:
            return TaskState.OPEN_CHEST
        return TaskState.GO_TO_EXIT

    if state == TaskState.KILL_GUARDIAN:
        if not symbol_map.monsters:
            return TaskState.GO_TO_EXIT
        if not has_sword:
            return TaskState.OPEN_CHEST if symbol_map.chests else TaskState.GO_TO_EXIT
        return TaskState.KILL_GUARDIAN

    if state == TaskState.PRESS_BUTTON:
        if symbol_map.buttons or symbol_map.switches:
            return TaskState.PRESS_BUTTON
        return TaskState.GO_TO_EXIT

    if state == TaskState.GO_TO_EXIT:
        
        if not symbol_map.exits and symbol_map.chests and key_count <= 0:
            return TaskState.GET_KEY
        return TaskState.GO_TO_EXIT

    return state


def _nearest(player: GridPos | None, targets) -> Optional[GridPos]:
    target_tuple = tuple(targets)
    if not target_tuple:
        return None
    if player is None:
        return target_tuple[0]
    return min(target_tuple, key=lambda pos: _manhattan(player, pos))


def _manhattan(left: GridPos, right: GridPos) -> int:
    return abs(left[0] - right[0]) + abs(left[1] - right[1])


def _inventory_int(inventory: dict, *names: str) -> int:
    for name in names:
        value = inventory.get(name)
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, (int, float)):
            return int(value)
        if isinstance(value, str) and value.isdigit():
            return int(value)
    return 0


def _inventory_bool(inventory: dict, *names: str) -> bool:
    for name in names:
        value = inventory.get(name)
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value > 0
        if isinstance(value, str):
            return value.lower() in {"1", "true", "yes", "y", name}
    for container_name in ("items", "tools"):
        values = inventory.get(container_name, ())
        if isinstance(values, (list, tuple, set)) and any(str(value) in names for value in values):
            return True
    return False
