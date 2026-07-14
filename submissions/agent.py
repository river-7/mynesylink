from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from nesylink.core.constants import (
    ACTION_A,
    ACTION_B,
    ACTION_DOWN,
    ACTION_LEFT,
    ACTION_NOOP,
    ACTION_RIGHT,
    ACTION_UP,
    GRID_HEIGHT,
    GRID_WIDTH,
    TILE_SIZE,
)
from submissions.fsm import TaskState, get_goal, get_initial_state, update_state
from submissions.planner import a_star, build_occupancy, path_to_actions, plan_path
from submissions.vision import VisionState, normalize_agent_observation


GridPos = tuple[int, int]

_ACTION_TO_DELTA: dict[int, GridPos] = {
    ACTION_UP: (0, -1),
    ACTION_DOWN: (0, 1),
    ACTION_LEFT: (-1, 0),
    ACTION_RIGHT: (1, 0),
}


class TaskAgent:
    """Integrated policy that connects vision, FSM, and path planning."""

    def __init__(self, task_id: str):
        self.task_id = task_id
        self.vision = VisionState()
        self.state = get_initial_state(task_id)
        self.last_reward = 0.0
        self._move_action = ACTION_NOOP
        self._move_ticks_remaining = 0
        self._move_start_player: GridPos | None = None
        self._last_player: GridPos | None = None
        self._interaction_cooldown = 0
        self._failed_tiles: dict[GridPos, int] = {}
        self._known_exits: set[GridPos] = set()
        self._locked_exit: GridPos | None = None
        self._recovery_actions: list[int] = []
        self._settle_ticks_remaining = 0
        self._last_room_jump = False
        self._entry_side: str | None = None
        self._switch_stages_pressed: set[tuple[bool, bool]] = set()
        self._opened_chests: set[GridPos] = set()
        self._pending_chest_target: GridPos | None = None
        self._pending_chest_open: GridPos | None = None
        self._pending_attack_target: GridPos | None = None
        self._pressed_buttons: set[GridPos] = set()
        self._used_exit_sides_by_room: dict[tuple[GridPos, ...], set[str]] = {}
        self._blocked_exit_sides_by_room: dict[tuple[GridPos, ...], set[str]] = {}

    def step(self, frame: np.ndarray, inventory: dict) -> int:
        symbol_map = self.vision.observe(frame, reward=self.last_reward)
        self.state = update_state(self.state, symbol_map, inventory)
        self._decay_failed_tiles()

        player = symbol_map.player if symbol_map.player is not None else self._last_player
        if player is None or self.state == TaskState.DONE:
            return ACTION_NOOP
        self._reconcile_pending_chest(symbol_map, inventory)
        self._reconcile_visible_chests(symbol_map, inventory)

        if symbol_map.player is not None:
            self._last_room_jump = False
            if self._last_player is not None and _manhattan(symbol_map.player, self._last_player) > 2:
                self._move_ticks_remaining = 0
                self._last_room_jump = True
                self._move_action = ACTION_NOOP
                self._move_start_player = None
                self._settle_ticks_remaining = 0
                self._recovery_actions = []
                self._failed_tiles = {}
                self._known_exits = set()
                self._locked_exit = None
                self._entry_side = _boundary_side(symbol_map.player)
            self._last_player = symbol_map.player
        if self.last_reward <= -0.05 and self._move_action != ACTION_NOOP:
            blocked_action = self._move_action
            failed_tile = _step(player, blocked_action)
            recovery = self._recovery_for_failed_move(symbol_map, player, blocked_action)
            if recovery:
                self._recovery_actions = recovery
            elif _in_bounds(failed_tile):
                self._failed_tiles[failed_tile] = 80
            self._move_ticks_remaining = 0
            self._move_action = ACTION_NOOP
            self._move_start_player = None
            self._settle_ticks_remaining = 0
        if player in set(symbol_map.buttons):
            self._pressed_buttons.add(player)
        self._apply_exploration_state(symbol_map, inventory)
        self._known_exits.update(symbol_map.exits)
        self._record_blocked_exit_feedback(symbol_map, inventory, player)

        defensive_action = self._defensive_action(symbol_map, inventory, player)
        if defensive_action is not None:
            return defensive_action

        # Keep pushing outward after reaching any known boundary exit until the environment terminates.
        if self.state == TaskState.GO_TO_EXIT and self._player_on_known_exit(player):
            if self._entry_side is not None:
                desired_exit = self._exploration_route_goal(symbol_map, inventory, player)
                if desired_exit is not None and not self._player_on_exit_segment(player, {desired_exit} | set(symbol_map.exits)):
                    pass
                else:
                    self._move_ticks_remaining = 0
                    self._mark_exit_used(player, set(symbol_map.exits) | self._known_exits)
                    return _outward_action(player)
            else:
                self._move_ticks_remaining = 0
                self._mark_exit_used(player, set(symbol_map.exits) | self._known_exits)
                return _outward_action(player)
        if (
            self._move_action != ACTION_NOOP
            and self._move_start_player is not None
            and player != self._move_start_player
        ):
            if _movement_reached_boundary(self._move_action, player):
                self._move_ticks_remaining = 0
                self._move_action = ACTION_NOOP
                self._move_start_player = None
                self._settle_ticks_remaining = 0
            else:
                if self._settle_ticks_remaining <= 0:
                    self._settle_ticks_remaining = 4
                self._settle_ticks_remaining -= 1
                action = self._move_action
                if self._settle_ticks_remaining <= 0:
                    self._move_ticks_remaining = 0
                    self._move_action = ACTION_NOOP
                    self._move_start_player = None
                return action

        # Interactions take priority over continuing a movement plan.
        goal = get_goal(self.state, symbol_map)
        exploration_goal = self._exploration_route_goal(symbol_map, inventory, player)
        if exploration_goal is not None:
            goal = exploration_goal
        elif self.state == TaskState.GO_TO_EXIT:
            goal = self._locked_exit_goal(player, goal, symbol_map)
        visible_or_known_exits = set(symbol_map.exits) | self._known_exits
        goal_exit_component = _exit_component(visible_or_known_exits, goal) if goal in visible_or_known_exits else set()
        if goal in visible_or_known_exits and self._player_on_exit_segment(player, goal_exit_component):
            self._move_ticks_remaining = 0
            self._move_action = ACTION_NOOP
            self._move_start_player = None
            self._mark_exit_used(player, goal_exit_component)
            return _outward_action(player)
        should_press_switch = self._should_press_switch(symbol_map, inventory)
        if should_press_switch and goal in set(symbol_map.switches) and _manhattan(player, goal) == 1:
            self._record_switch_press(inventory)
            self._move_ticks_remaining = 0
            self._interaction_cooldown = 4
            return ACTION_A
        should_open_chest = self.state in {TaskState.GET_KEY, TaskState.OPEN_CHEST}
        if should_open_chest and goal in self._available_chests(symbol_map) and _manhattan(player, goal) == 1:
            self._move_ticks_remaining = 0
            self._move_action = ACTION_NOOP
            self._move_start_player = None
            self._recovery_actions = []
            return self._face_then_open_chest(player, goal)
        interaction = self._interaction_action(
            symbol_map,
            inventory,
            player=player,
            allow_chests=should_open_chest,
            allow_switches=should_press_switch,
        )
        if interaction is not None:
            self._move_ticks_remaining = 0
            return interaction

        self._pending_chest_target = None
        if self._interaction_cooldown > 0:
            self._interaction_cooldown -= 1
            return ACTION_NOOP

        if self._recovery_actions:
            action = self._recovery_actions.pop(0)
            return action

        if self._move_start_player is not None and self._move_action != ACTION_NOOP and self._move_ticks_remaining <= 0:
            if player == self._move_start_player:
                failed_tile = _step(player, self._move_action)
                recovery = self._recovery_for_failed_move(symbol_map, player, self._move_action)
                if recovery:
                    self._recovery_actions = recovery[1:]
                    self._move_action = ACTION_NOOP
                    self._move_start_player = None
                    return recovery[0]
                if _in_bounds(failed_tile):
                    self._failed_tiles[failed_tile] = 80
            self._move_action = ACTION_NOOP
            self._move_start_player = None

        # Pixel movement needs the same direction repeated for roughly one tile.
        if self._move_ticks_remaining > 0 and self._move_action != ACTION_NOOP:
            if self._move_start_player is not None and player != self._move_start_player:
                self._move_ticks_remaining = 0
                self._move_action = ACTION_NOOP
                self._move_start_player = None
            else:
                self._move_ticks_remaining -= 1
                return self._move_action

        if goal is None:
            return ACTION_NOOP

        planned_goal = goal
        if self._needs_adjacent_planning(symbol_map, goal):
            planned_goal = self._adjacent_goal(symbol_map, goal) or goal

        actions = self._plan_actions(symbol_map, player, planned_goal)
        if not actions:
            return ACTION_NOOP

        action = actions[0]
        self._move_action = action
        self._move_ticks_remaining = max(TILE_SIZE - 1, 0)
        self._move_start_player = player

        return action

    def update_reward(self, reward: float) -> None:
        self.last_reward = float(reward)

    def reset(self) -> None:
        self.vision.reset()
        self.state = get_initial_state(self.task_id)
        self.last_reward = 0.0
        self._move_action = ACTION_NOOP
        self._move_ticks_remaining = 0
        self._move_start_player = None
        self._last_player = None
        self._interaction_cooldown = 0
        self._failed_tiles = {}
        self._known_exits = set()
        self._locked_exit = None
        self._recovery_actions = []
        self._settle_ticks_remaining = 0
        self._last_room_jump = False
        self._entry_side = None
        self._switch_stages_pressed = set()
        self._opened_chests = set()
        self._pending_chest_target = None
        self._pending_chest_open = None
        self._pending_attack_target = None
        self._pressed_buttons = set()
        self._used_exit_sides_by_room = {}
        self._blocked_exit_sides_by_room = {}

    def _interaction_action(
        self,
        symbol_map,
        inventory: dict,
        player: GridPos | None = None,
        *,
        allow_chests: bool = True,
        allow_switches: bool = True,
    ) -> int | None:
        player = symbol_map.player if symbol_map.player is not None else player
        if player is None:
            return None

        targets: tuple[GridPos, ...] = ()
        if allow_switches:
            targets = targets + tuple(symbol_map.switches)
        if allow_chests:
            targets = tuple(self._available_chests(symbol_map)) + targets
        for target in targets:
            if _manhattan(player, target) <= 1:
                if target in self._available_chests(symbol_map):
                    self._move_action = ACTION_NOOP
                    self._move_start_player = None
                    self._recovery_actions = []
                    return self._face_then_open_chest(player, target)
                if target in set(symbol_map.switches):
                    self._record_switch_press(inventory)
                self._interaction_cooldown = 2
                return ACTION_A

        if _has_inventory_item(inventory, "sword", "has_sword") and symbol_map.monsters:
            monster = min(symbol_map.monsters, key=lambda pos: _manhattan(player, pos))
            if _manhattan(player, monster) <= 1:
                return self._face_then_attack(player, monster, inventory)
        return None

    def _face_then_open_chest(self, player: GridPos, target: GridPos) -> int:
        face_action = _action_between(player, target)
        if self._pending_chest_target == target:
            self._pending_chest_target = None
            self._pending_chest_open = target
            self._interaction_cooldown = 2
            return ACTION_A
        self._pending_chest_target = target
        return face_action if face_action != ACTION_NOOP else ACTION_A

    def _reconcile_pending_chest(self, symbol_map, inventory: dict) -> None:
        if self._pending_chest_open is None:
            return
        target = self._pending_chest_open
        opened_by_feedback = (
            self.last_reward > 0.5
            or _inventory_key_count(inventory) > 0
            or _has_inventory_item(inventory, "sword", "has_sword")
            or target not in set(symbol_map.chests)
        )
        if opened_by_feedback:
            self._opened_chests.add(target)
        self._pending_chest_open = None

    def _reconcile_visible_chests(self, symbol_map, inventory: dict) -> None:
        if self.state != TaskState.GET_KEY or _inventory_key_count(inventory) > 0:
            return
        visible_chests = set(symbol_map.chests)
        if not visible_chests:
            return
        self._opened_chests.difference_update(visible_chests)

    def _defensive_action(self, symbol_map, inventory: dict, player: GridPos) -> int | None:
        if not symbol_map.monsters:
            self._pending_attack_target = None
            return None
        nearest = min(symbol_map.monsters, key=lambda pos: _manhattan(player, pos))
        distance = _manhattan(player, nearest)
        if distance > 1:
            self._pending_attack_target = None
            return None
        self._move_ticks_remaining = 0
        self._move_action = ACTION_NOOP
        self._move_start_player = None
        self._recovery_actions = []
        if _has_inventory_item(inventory, "sword", "has_sword"):
            return self._face_then_attack(player, nearest, inventory)
        return ACTION_B

    def _face_then_attack(self, player: GridPos, target: GridPos, inventory: dict) -> int:
        face_action = _action_between(player, target)
        if self._pending_attack_target == target:
            self._pending_attack_target = None
            self._interaction_cooldown = 2
            return ACTION_A
        self._pending_attack_target = target
        return face_action if face_action != ACTION_NOOP else ACTION_A

    def _apply_exploration_state(self, symbol_map, inventory: dict) -> None:
        if self._available_chests(symbol_map):
            self.state = TaskState.GET_KEY
            return
        if (
            self.state == TaskState.KILL_GUARDIAN
            and symbol_map.monsters
            and _has_inventory_item(inventory, "sword", "has_sword")
        ):
            self.state = TaskState.KILL_GUARDIAN
            return
        if _inventory_key_count(inventory) > 0:
            return
        if self.state == TaskState.GO_TO_EXIT:
            self.state = TaskState.GET_KEY

    def _exploration_route_goal(self, symbol_map, inventory: dict, player: GridPos) -> GridPos | None:
        available_chests = self._available_chests(symbol_map)
        if available_chests:
            return min(available_chests, key=lambda pos: _manhattan(player, pos))
        button_goal = self._button_goal(symbol_map, player)
        if button_goal is not None:
            return button_goal
        if self._should_press_switch(symbol_map, inventory):
            return min(symbol_map.switches, key=lambda pos: _manhattan(player, pos))
        key_count = _inventory_key_count(inventory)
        if key_count > 0:
            if self._entry_side is not None and symbol_map.exits:
                return self._choose_room_exit(symbol_map, player, prefer_unvisited=self.state == TaskState.GO_TO_EXIT)
            if self.state != TaskState.GO_TO_EXIT or not symbol_map.exits:
                return None
            return self._choose_room_exit(symbol_map, player, prefer_unvisited=True)
        if not symbol_map.exits:
            if self.state == TaskState.KILL_GUARDIAN and symbol_map.monsters and _has_inventory_item(inventory, "sword", "has_sword"):
                return min(symbol_map.monsters, key=lambda pos: _manhattan(player, pos))
            return None
        exit_goal = self._choose_room_exit(symbol_map, player)
        if exit_goal is not None:
            return exit_goal
        if self.state == TaskState.KILL_GUARDIAN and symbol_map.monsters and _has_inventory_item(inventory, "sword", "has_sword"):
            return min(symbol_map.monsters, key=lambda pos: _manhattan(player, pos))
        return None

    def _available_chests(self, symbol_map) -> tuple[GridPos, ...]:
        return tuple(chest for chest in symbol_map.chests if chest not in self._opened_chests)

    def _choose_room_exit(self, symbol_map, player: GridPos, *, prefer_unvisited: bool = False) -> GridPos | None:
        exits = tuple(symbol_map.exits)
        if not exits:
            return None
        if prefer_unvisited:
            candidates = list(exits)
            excluded_entry_side = False
        else:
            excluded_entry_side = any(_boundary_side(exit_pos) != self._entry_side for exit_pos in exits)
            candidates = [exit_pos for exit_pos in exits if _boundary_side(exit_pos) != self._entry_side]
            if not candidates:
                candidates = list(exits)
        blocked_sides = self._blocked_exit_sides_for_exits(set(exits))
        if blocked_sides:
            unblocked = [exit_pos for exit_pos in candidates if _boundary_side(exit_pos) not in blocked_sides]
            if unblocked:
                candidates = unblocked
        reachable = [exit_pos for exit_pos in candidates if self._exit_is_reachable(symbol_map, player, exit_pos)]
        if not reachable and excluded_entry_side:
            fallback = [exit_pos for exit_pos in exits if _boundary_side(exit_pos) not in blocked_sides]
            if not fallback:
                fallback = list(exits)
            reachable = [exit_pos for exit_pos in fallback if self._exit_is_reachable(symbol_map, player, exit_pos)]
        if reachable:
            return min(reachable, key=lambda pos: self._exit_choice_priority(pos, player, symbol_map, prefer_unvisited))
        return min(candidates, key=lambda pos: self._exit_choice_priority(pos, player, symbol_map, prefer_unvisited))

    def _exit_choice_priority(
        self,
        exit_pos: GridPos,
        player: GridPos,
        symbol_map,
        prefer_unvisited: bool,
    ) -> tuple[int, ...]:
        if not prefer_unvisited:
            return (0, _manhattan(player, exit_pos))
        side = _boundary_side(exit_pos)
        used_sides = self._used_exit_sides_for_exits(set(symbol_map.exits))
        has_unvisited = any(_boundary_side(candidate) not in used_sides for candidate in symbol_map.exits)
        used_penalty = 1 if has_unvisited and side in used_sides else 0
        entry_penalty = 1 if side == self._entry_side else 0
        return (used_penalty, entry_penalty, _manhattan(player, exit_pos))

    def _mark_exit_used(self, player: GridPos, exits: set[GridPos]) -> None:
        side = _boundary_side(player)
        if side is None:
            return
        signature = self._room_exit_signature(exits)
        if not signature:
            return
        self._used_exit_sides_by_room.setdefault(signature, set()).add(side)

    def _record_blocked_exit_feedback(self, symbol_map, inventory: dict, player: GridPos) -> None:
        if _inventory_key_count(inventory) > 0:
            self._blocked_exit_sides_by_room.clear()
            return
        if self.last_reward > -0.05 or self._move_action != ACTION_NOOP:
            return
        side = _boundary_side(player)
        if side is None:
            return
        exits = set(symbol_map.exits)
        if not any(_boundary_side(exit_pos) == side for exit_pos in exits):
            return
        signature = self._room_exit_signature(exits)
        if not signature:
            return
        self._blocked_exit_sides_by_room.setdefault(signature, set()).add(side)

    def _used_exit_sides_for_exits(self, exits: set[GridPos]) -> set[str]:
        signature = set(self._room_exit_signature(exits))
        used: set[str] = set()
        for known_signature, sides in self._used_exit_sides_by_room.items():
            known_exits = set(known_signature)
            if known_exits == signature or known_exits.issubset(signature):
                used.update(sides)
        return used

    def _blocked_exit_sides_for_exits(self, exits: set[GridPos]) -> set[str]:
        signature = set(self._room_exit_signature(exits))
        blocked: set[str] = set()
        for known_signature, sides in self._blocked_exit_sides_by_room.items():
            known_exits = set(known_signature)
            if known_exits == signature or known_exits.issubset(signature) or bool(known_exits & signature):
                blocked.update(sides)
        return blocked

    def _room_exit_signature(self, exits: set[GridPos]) -> tuple[GridPos, ...]:
        return tuple(sorted(exits))

    def _button_goal(self, symbol_map, player: GridPos) -> GridPos | None:
        buttons = tuple(button for button in symbol_map.buttons if button not in self._pressed_buttons)
        if not buttons:
            return None
        unpressed_like = [button for button in buttons if button != player]
        if not unpressed_like:
            return None
        reachable = [button for button in unpressed_like if self._tile_is_reachable(symbol_map, player, button)]
        if reachable:
            return min(reachable, key=lambda pos: _manhattan(player, pos))
        return min(unpressed_like, key=lambda pos: _manhattan(player, pos))

    def _should_press_switch(self, symbol_map, inventory: dict) -> bool:
        if not symbol_map.switches:
            return False
        key_count = _inventory_key_count(inventory)
        has_sword = _has_inventory_item(inventory, "sword", "has_sword")
        if key_count <= 0 and not has_sword:
            return False
        return self._switch_stage(inventory) not in self._switch_stages_pressed

    def _switch_stage(self, inventory: dict) -> tuple[bool, bool]:
        return (_inventory_key_count(inventory) > 0, _has_inventory_item(inventory, "sword", "has_sword"))

    def _record_switch_press(self, inventory: dict) -> None:
        self._switch_stages_pressed.add(self._switch_stage(inventory))

    def _tile_is_reachable(self, symbol_map, player: GridPos, goal: GridPos) -> bool:
        if player == goal:
            return True
        return bool(plan_path(symbol_map, player, goal))

    def _exit_is_reachable(self, symbol_map, player: GridPos, exit_pos: GridPos) -> bool:
        if self._player_on_exit_segment(player, {exit_pos}):
            return True
        for approach in self._exit_approaches(symbol_map, exit_pos):
            if player == approach or plan_path(symbol_map, player, approach):
                return True
        for exit_tile in _exit_component(set(symbol_map.exits) | {exit_pos}, exit_pos):
            if player == exit_tile or plan_path(symbol_map, player, exit_tile):
                return True
        return bool(plan_path(symbol_map, player, exit_pos))

    def _needs_adjacent_planning(self, symbol_map, goal: GridPos) -> bool:
        interactive_targets = (
            set(symbol_map.chests)
            | set(symbol_map.monsters)
            | set(symbol_map.switches)
        )
        return goal in interactive_targets

    def _same_exit_segment(self, player: GridPos, exit_pos: GridPos) -> bool:
        if player == exit_pos:
            return True
        if player[1] == exit_pos[1] and player[1] in (0, GRID_HEIGHT - 1):
            return abs(player[0] - exit_pos[0]) <= 1
        if player[0] == exit_pos[0] and player[0] in (0, GRID_WIDTH - 1):
            return abs(player[1] - exit_pos[1]) <= 1
        return False

    def _player_on_exit_segment(self, player: GridPos, exits: set[GridPos]) -> bool:
        if player in exits:
            return True
        return False

    def _adjacent_goal(self, symbol_map, target: GridPos) -> GridPos | None:
        player = symbol_map.player
        if player is None:
            return None

        blocked = set(symbol_map.blocked_tiles())
        danger = set(symbol_map.danger_tiles())
        candidates: list[GridPos] = []
        for dx, dy in _ACTION_TO_DELTA.values():
            candidate = (target[0] + dx, target[1] + dy)
            if not _in_bounds(candidate):
                continue
            if candidate in blocked:
                continue
            if candidate in danger and self.state != TaskState.KILL_GUARDIAN:
                continue
            candidates.append(candidate)

        if not candidates:
            return None
        if player in candidates:
            return player

        reachable: list[tuple[int, GridPos]] = []
        for candidate in sorted(candidates, key=lambda pos: self._adjacent_goal_priority(player, target, pos)):
            actions = plan_path(symbol_map, player, candidate)
            if actions:
                priority = self._adjacent_goal_priority(player, target, candidate)
                reachable.append((len(actions) + priority, candidate))
        if reachable:
            return min(reachable, key=lambda item: (item[0], _manhattan(player, item[1])))[1]
        return min(candidates, key=lambda pos: self._adjacent_goal_priority(player, target, pos))

    def _adjacent_goal_priority(self, player: GridPos, target: GridPos, candidate: GridPos) -> int:
        """Prefer adjacent interaction cells that avoid pixel-level corner clipping."""

        priority = _manhattan(player, candidate)
        diagonal_to_target = abs(player[0] - target[0]) == 1 and abs(player[1] - target[1]) == 1
        if diagonal_to_target:
            clips_corner = candidate == (target[0], player[1])
            clears_corner = candidate == (player[0], target[1])
            if clips_corner:
                priority += 10
            elif clears_corner:
                priority -= 2
        return priority

    def _plan_actions(self, symbol_map, start: GridPos, goal: GridPos) -> list[int]:
        if self.state == TaskState.GO_TO_EXIT or goal in set(symbol_map.exits):
            return self._plan_to_exit(symbol_map, start, goal)
        if self._failed_tiles:
            return self._plan_avoiding_failed_tiles(symbol_map, start, goal)
        return plan_path(symbol_map, start, goal)

    def _plan_avoiding_failed_tiles(self, symbol_map, start: GridPos, goal: GridPos) -> list[int]:
        occupancy = build_occupancy(symbol_map)
        occupancy[start[1], start[0]] = 0

        for tile in self._failed_tiles:
            if _in_bounds(tile) and tile != goal:
                occupancy[tile[1], tile[0]] = 1

        blocked = set(symbol_map.blocked_tiles()) | set(symbol_map.monsters)
        if goal not in blocked:
            occupancy[goal[1], goal[0]] = 0
            path = a_star(occupancy, start, goal)
            return path_to_actions([start, *path]) if path else []

        best_path: list[GridPos] = []
        for adjacent in _neighbors(goal):
            if occupancy[adjacent[1], adjacent[0]] != 0:
                continue
            path = a_star(occupancy, start, adjacent)
            if path and (not best_path or len(path) < len(best_path)):
                best_path = path
        return path_to_actions([start, *best_path]) if best_path else []

    def _plan_to_exit(self, symbol_map, start: GridPos, goal: GridPos) -> list[int]:
        occupancy = build_occupancy(symbol_map)
        for danger in set(symbol_map.danger_tiles()):
            if _in_bounds(danger) and danger not in {start, goal}:
                occupancy[danger[1], danger[0]] = 1
        for failed in self._failed_tiles:
            if _in_bounds(failed) and failed not in {start, goal}:
                occupancy[failed[1], failed[0]] = 1
        occupancy[start[1], start[0]] = 0
        occupancy[goal[1], goal[0]] = 0

        approaches = self._exit_approaches(symbol_map, goal)
        if approaches:
            for approach in approaches:
                occupancy[approach[1], approach[0]] = 0
            if start in approaches:
                return [_outward_action(goal)]
            best_path: list[GridPos] = []
            for approach in sorted(approaches, key=lambda pos: _manhattan(start, pos)):
                path = a_star(occupancy, start, approach)
                if path and (not best_path or len(path) < len(best_path)):
                    best_path = path
            if best_path:
                return path_to_actions([start, *best_path])

        exit_component = _exit_component(set(symbol_map.exits) | {goal}, goal)
        for exit_tile in exit_component:
            if _in_bounds(exit_tile):
                occupancy[exit_tile[1], exit_tile[0]] = 0
        if start in exit_component:
            return [_outward_action(start)]
        best_exit_path: list[GridPos] = []
        for exit_tile in sorted(exit_component, key=lambda pos: _manhattan(start, pos)):
            if not _in_bounds(exit_tile):
                continue
            path = a_star(occupancy, start, exit_tile)
            if path and (not best_exit_path or len(path) < len(best_exit_path)):
                best_exit_path = path
        if best_exit_path:
            return path_to_actions([start, *best_exit_path])

        approach = _inside_exit_approach(goal)
        if approach is not None and _in_bounds(approach):
            if start == approach:
                return [_outward_action(goal)]
            path_to_approach = a_star(occupancy, start, approach)
            if path_to_approach:
                return path_to_actions([start, *path_to_approach])

        path = a_star(occupancy, start, goal)
        return path_to_actions([start, *path]) if path else []

    def _exit_approaches(self, symbol_map, goal: GridPos) -> tuple[GridPos, ...]:
        exits = set(symbol_map.exits)
        if goal not in exits:
            exits.add(goal)
        component = _exit_component(exits, goal)
        blocked = set(symbol_map.blocked_tiles())
        danger = set(symbol_map.danger_tiles())
        approaches: list[GridPos] = []
        for exit_pos in sorted(component):
            approach = _inside_exit_approach(exit_pos)
            if approach is None or not _in_bounds(approach):
                continue
            if approach in blocked or approach in danger:
                continue
            if approach not in approaches:
                approaches.append(approach)
        return tuple(approaches)

    def _recovery_for_failed_move(self, symbol_map, player: GridPos, action: int) -> list[int]:
        failed_tile = _step(player, action)
        if not _in_bounds(failed_tile):
            return []

        blocked = set(symbol_map.blocked_tiles()) | set(symbol_map.monsters)
        exit_tiles = set(symbol_map.exits) | self._known_exits
        if self.state == TaskState.GO_TO_EXIT:
            exit_outward = _nearby_exit_outward_action(failed_tile, exit_tiles, action)
            if exit_outward != ACTION_NOOP:
                nudge_ticks = max(6, TILE_SIZE // 2)
                retry_ticks = TILE_SIZE
                return [exit_outward] * nudge_ticks + [action] * retry_ticks

        nudge_actions: list[int] = []
        if action in (ACTION_UP, ACTION_DOWN):
            left_side = (failed_tile[0] - 1, failed_tile[1])
            right_side = (failed_tile[0] + 1, failed_tile[1])
            left_blocked = (not _in_bounds(left_side)) or left_side in blocked
            right_blocked = (not _in_bounds(right_side)) or right_side in blocked
            if left_blocked and not right_blocked:
                nudge_actions = [ACTION_RIGHT, ACTION_LEFT]
            elif right_blocked and not left_blocked:
                nudge_actions = [ACTION_LEFT, ACTION_RIGHT]
            else:
                nudge_actions = [ACTION_RIGHT, ACTION_LEFT]
        elif action in (ACTION_LEFT, ACTION_RIGHT):
            upper_side = (failed_tile[0], failed_tile[1] - 1)
            lower_side = (failed_tile[0], failed_tile[1] + 1)
            upper_blocked = (not _in_bounds(upper_side)) or upper_side in blocked
            lower_blocked = (not _in_bounds(lower_side)) or lower_side in blocked
            if upper_blocked and not lower_blocked:
                nudge_actions = [ACTION_DOWN, ACTION_UP]
            elif lower_blocked and not upper_blocked:
                nudge_actions = [ACTION_UP, ACTION_DOWN]
            else:
                nudge_actions = [ACTION_DOWN, ACTION_UP]

        nudge_action = ACTION_NOOP
        allow_blocked_micro_nudge = False
        if nudge_actions:
            first_choice = nudge_actions[0]
            if _can_step(player, first_choice, blocked):
                nudge_action = first_choice
            else:
                # A blocked neighbouring tile can still allow a few pixels of
                # in-tile correction. This is exactly the corner-clip case near
                # chests/walls: crossing a full tile is illegal, but nudging
                # away from the clipping edge can make the retry pass.
                nudge_action = first_choice
                allow_blocked_micro_nudge = True
        if nudge_action == ACTION_NOOP and len(nudge_actions) > 1:
            nudge_action = next(
                (candidate for candidate in nudge_actions[1:] if _can_step(player, candidate, blocked)),
                ACTION_NOOP,
            )
        if nudge_action == ACTION_NOOP:
            return []
        nudge_ticks = max(3, TILE_SIZE // 4) if allow_blocked_micro_nudge else max(4, TILE_SIZE // 3)
        retry_ticks = TILE_SIZE
        return [nudge_action] * nudge_ticks + [action] * retry_ticks

    def _locked_exit_goal(self, player: GridPos, fallback: GridPos | None, symbol_map) -> GridPos | None:
        exits = self._known_exits or ({fallback} if fallback is not None else set())
        exits = {exit_pos for exit_pos in exits if exit_pos is not None}
        if self._locked_exit in exits and self._exit_has_open_approach(self._locked_exit, symbol_map):
            return self._locked_exit
        if not exits:
            self._locked_exit = fallback
            return fallback
        open_exits = [exit_pos for exit_pos in exits if self._exit_has_open_approach(exit_pos, symbol_map)]
        candidates = open_exits or list(exits)
        self._locked_exit = min(candidates, key=lambda pos: _manhattan(player, pos))
        return self._locked_exit

    def _exit_has_open_approach(self, exit_pos: GridPos | None, symbol_map) -> bool:
        if exit_pos is None:
            return False
        approach = _inside_exit_approach(exit_pos)
        if approach is None:
            return True
        return _in_bounds(approach) and approach not in set(symbol_map.blocked_tiles())

    def _player_on_known_exit(self, player: GridPos) -> bool:
        if self._player_on_exit_segment(player, self._known_exits):
            return True
        if self._locked_exit is None:
            return False
        return self._player_on_exit_segment(player, {self._locked_exit})

    def _decay_failed_tiles(self) -> None:
        expired: list[GridPos] = []
        for tile, ttl in self._failed_tiles.items():
            next_ttl = ttl - 1
            if next_ttl <= 0:
                expired.append(tile)
            else:
                self._failed_tiles[tile] = next_ttl
        for tile in expired:
            del self._failed_tiles[tile]



def get_action_mask(symbol_map, inventory: dict | None = None) -> list[bool]:
    """Return a length-7 action mask for formal checks."""

    del inventory
    mask = [True] * 7
    player = symbol_map.player
    if player is None:
        return [True, False, False, False, False, False, False]

    blocked = set(symbol_map.blocked_tiles())
    for action, (dx, dy) in _ACTION_TO_DELTA.items():
        nxt = (player[0] + dx, player[1] + dy)
        if not _in_bounds(nxt) or nxt in blocked:
            mask[action] = False
    adjacent_interact = any(
        _manhattan(player, target) <= 1
        for target in tuple(symbol_map.chests) + tuple(symbol_map.buttons) + tuple(symbol_map.switches)
    )
    adjacent_monster = any(_manhattan(player, monster) <= 1 for monster in symbol_map.monsters)
    mask[ACTION_A] = adjacent_interact or adjacent_monster
    return mask


def run_episode(env, task_id: str, max_steps: int = 2000) -> None:
    agent = TaskAgent(task_id)
    obs, info = env.reset()

    for _ in range(1, max_steps + 1):
        frame = _extract_frame(obs, env)
        inventory = _normalize_inventory(info.get("inventory", {})) if isinstance(info, dict) else {}
        action = agent.step(frame, inventory)
        obs, reward, terminated, truncated, info = env.step(action)
        agent.update_reward(float(reward))
        if terminated or truncated:
            return


def _extract_frame(obs: Any, env: Any | None = None) -> np.ndarray:
    if isinstance(obs, dict):
        if "frame" in obs or "obs" in obs:
            normalized = normalize_agent_observation(obs)
            return normalized.frame
        if env is not None:
            return np.asarray(env.render())
        raise KeyError("dict observation must contain 'frame' or 'obs'")
    return np.asarray(obs)


def _normalize_inventory(inventory: Any) -> dict:
    if isinstance(inventory, dict):
        return inventory
    if inventory is None:
        return {}

    result: dict[str, Any] = {}
    for item in _as_iterable(inventory):
        if isinstance(item, str):
            result[item] = result.get(item, 0) + 1
    return result


def _as_iterable(value: Any) -> Iterable[Any]:
    if isinstance(value, (str, bytes)):
        return (value,)
    try:
        return iter(value)
    except TypeError:
        return ()


def _has_inventory_item(inventory: dict, *names: str) -> bool:
    for name in names:
        value = inventory.get(name)
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value > 0
    for container_name in ("items", "tools"):
        values = inventory.get(container_name, ())
        if isinstance(values, (list, tuple, set)) and any(str(value) in names for value in values):
            return True
    return False


def _inventory_key_count(inventory: dict) -> int:
    for name in ("keys", "key"):
        value = inventory.get(name)
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, (int, float)):
            return int(value)
        if isinstance(value, str) and value.isdigit():
            return int(value)
    return 0


def _player_on_exit(symbol_map) -> bool:
    return symbol_map.player is not None and symbol_map.player in set(symbol_map.exits)


def _outward_action(pos: GridPos) -> int:
    x, y = pos
    if y == 0:
        return ACTION_UP
    if y == GRID_HEIGHT - 1:
        return ACTION_DOWN
    if x == 0:
        return ACTION_LEFT
    if x == GRID_WIDTH - 1:
        return ACTION_RIGHT
    return ACTION_NOOP


def _boundary_side(pos: GridPos | None) -> str | None:
    if pos is None:
        return None
    x, y = pos
    if x <= 1:
        return "left"
    if x >= GRID_WIDTH - 2:
        return "right"
    if y <= 1:
        return "top"
    if y >= GRID_HEIGHT - 2:
        return "bottom"
    return None


def _inside_exit_approach(exit_pos: GridPos) -> GridPos | None:
    x, y = exit_pos
    if y == 0:
        return (x, 1)
    if y == GRID_HEIGHT - 1:
        return (x, GRID_HEIGHT - 2)
    if x == 0:
        return (1, y)
    if x == GRID_WIDTH - 1:
        return (GRID_WIDTH - 2, y)
    return None


def _movement_reached_boundary(action: int, pos: GridPos) -> bool:
    x, y = pos
    return (
        (action == ACTION_UP and y == 0)
        or (action == ACTION_DOWN and y == GRID_HEIGHT - 1)
        or (action == ACTION_LEFT and x == 0)
        or (action == ACTION_RIGHT and x == GRID_WIDTH - 1)
    )


def _nearby_exit_outward_action(failed_tile: GridPos, exits: set[GridPos], action: int) -> int:
    if action in (ACTION_LEFT, ACTION_RIGHT):
        if failed_tile in exits:
            outward = _outward_action(failed_tile)
            return outward if outward in (ACTION_UP, ACTION_DOWN) else ACTION_NOOP
        for exit_tile in exits:
            outward = _outward_action(exit_tile)
            if outward == ACTION_UP and failed_tile == (exit_tile[0], exit_tile[1] + 1):
                return ACTION_UP
            if outward == ACTION_DOWN and failed_tile == (exit_tile[0], exit_tile[1] - 1):
                return ACTION_DOWN
    if action in (ACTION_UP, ACTION_DOWN):
        if failed_tile in exits:
            outward = _outward_action(failed_tile)
            return outward if outward in (ACTION_LEFT, ACTION_RIGHT) else ACTION_NOOP
        for exit_tile in exits:
            outward = _outward_action(exit_tile)
            if outward == ACTION_LEFT and failed_tile == (exit_tile[0] + 1, exit_tile[1]):
                return ACTION_LEFT
            if outward == ACTION_RIGHT and failed_tile == (exit_tile[0] - 1, exit_tile[1]):
                return ACTION_RIGHT
    return ACTION_NOOP


def _exit_component(exits: set[GridPos], origin: GridPos) -> set[GridPos]:
    if origin not in exits:
        return {origin}
    component: set[GridPos] = set()
    frontier = [origin]
    while frontier:
        current = frontier.pop()
        if current in component:
            continue
        component.add(current)
        for dx, dy in _ACTION_TO_DELTA.values():
            nxt = (current[0] + dx, current[1] + dy)
            if nxt in exits and nxt not in component:
                frontier.append(nxt)
    return component


def _action_between(start: GridPos, goal: GridPos) -> int:
    delta = (goal[0] - start[0], goal[1] - start[1])
    for action, action_delta in _ACTION_TO_DELTA.items():
        if action_delta == delta:
            return action
    return ACTION_NOOP


def _step(pos: GridPos, action: int) -> GridPos:
    dx, dy = _ACTION_TO_DELTA.get(action, (0, 0))
    return pos[0] + dx, pos[1] + dy


def _neighbors(pos: GridPos) -> Iterable[GridPos]:
    x, y = pos
    for dx, dy in _ACTION_TO_DELTA.values():
        nxt = (x + dx, y + dy)
        if _in_bounds(nxt):
            yield nxt


def _can_step(pos: GridPos, action: int, blocked: set[GridPos]) -> bool:
    nxt = _step(pos, action)
    return _in_bounds(nxt) and nxt not in blocked


def _in_bounds(pos: GridPos) -> bool:
    x, y = pos
    return 0 <= x < GRID_WIDTH and 0 <= y < GRID_HEIGHT


def _manhattan(left: GridPos, right: GridPos) -> int:
    return abs(left[0] - right[0]) + abs(left[1] - right[1])


def _is_diagonal(left: GridPos, right: GridPos) -> bool:
    return abs(left[0] - right[0]) == 1 and abs(left[1] - right[1]) == 1


__all__ = ["TaskAgent", "get_action_mask", "run_episode"]
