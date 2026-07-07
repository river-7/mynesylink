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

    def step(self, frame: np.ndarray, inventory: dict) -> int:
        symbol_map = self.vision.observe(frame, reward=self.last_reward)
        self.state = update_state(self.state, symbol_map, inventory)
        self._decay_failed_tiles()
        self._known_exits.update(symbol_map.exits)

        player = symbol_map.player if symbol_map.player is not None else self._last_player
        if player is None or self.state == TaskState.DONE:
            return ACTION_NOOP

        if symbol_map.player is not None:
            self._last_room_jump = False
            if self._last_player is not None and _manhattan(symbol_map.player, self._last_player) > 2:
                self._move_ticks_remaining = 0
                self._last_room_jump = True
                if self._is_task3():
                    self._move_action = ACTION_NOOP
                    self._move_start_player = None
                    self._settle_ticks_remaining = 0
                    self._recovery_actions = []
                    self._failed_tiles = {}
                    self._known_exits = set()
                    self._locked_exit = None
                    self._entry_side = _boundary_side(symbol_map.player)
            self._last_player = symbol_map.player
        if self._is_task3():
            self._apply_task3_key_search_state(symbol_map, inventory)
            self._known_exits.update(symbol_map.exits)

        # Keep pushing outward after reaching any known boundary exit until the environment terminates.
        if self.state == TaskState.GO_TO_EXIT and self._player_on_known_exit(player):
            if self._is_task3() and _inventory_key_count(inventory) > 0:
                desired_exit = self._task3_route_goal(symbol_map, inventory, player)
                if desired_exit is not None and not self._same_exit_segment(player, desired_exit):
                    pass
                else:
                    self._move_ticks_remaining = 0
                    return _outward_action(player)
            else:
                self._move_ticks_remaining = 0
                return _outward_action(player)
        if (
            self._move_action != ACTION_NOOP
            and self._move_start_player is not None
            and player != self._move_start_player
        ):
            if self._settle_ticks_remaining <= 0:
                self._settle_ticks_remaining = 4
            self._settle_ticks_remaining -= 1
            action = self._move_action
            print(
                f"DEBUG settle: action={action}, remaining={self._settle_ticks_remaining}, "
                f"from={self._move_start_player}, player={player}"
            )
            if self._settle_ticks_remaining <= 0:
                self._move_ticks_remaining = 0
                self._move_action = ACTION_NOOP
                self._move_start_player = None
            return action

        # Interactions take priority over continuing a movement plan.
        goal = get_goal(self.state, symbol_map)
        task3_route_goal = self._task3_route_goal(symbol_map, inventory, player)
        if task3_route_goal is not None:
            goal = task3_route_goal
        elif self.state == TaskState.GO_TO_EXIT:
            goal = self._locked_exit_goal(player, goal, symbol_map)
        print(f"DEBUG state_goal: state={self.state}, player={player}, goal={goal}, keys={inventory.get('keys', inventory.get('key', 0))}")
        if goal in set(symbol_map.exits) and self._same_exit_segment(player, goal):
            self._move_ticks_remaining = 0
            self._move_action = ACTION_NOOP
            self._move_start_player = None
            print(f"DEBUG portal outward: player={player}, goal={goal}, action={_outward_action(player)}")
            return _outward_action(player)
        should_open_chest = self.state in {TaskState.GET_KEY, TaskState.OPEN_CHEST}
        if should_open_chest and goal in set(symbol_map.chests) and _manhattan(player, goal) == 1:
            self._move_ticks_remaining = 0
            self._interaction_cooldown = 2
            print(f"DEBUG interact: adjacent chest goal={goal}, player={player}, action=A")
            print(f"DEBUG A-action: player={player}, goal={goal}, inventory before={inventory}")
            return ACTION_A
        interaction = self._interaction_action(symbol_map, inventory, player=player, allow_chests=should_open_chest)
        if interaction is not None:
            self._move_ticks_remaining = 0
            return interaction

        if self._interaction_cooldown > 0:
            self._interaction_cooldown -= 1
            return ACTION_NOOP

        if self._recovery_actions:
            action = self._recovery_actions.pop(0)
            print(f"DEBUG recovery: action={action}, remaining={len(self._recovery_actions)}, player={player}")
            return action

        if self._move_start_player is not None and self._move_action != ACTION_NOOP and self._move_ticks_remaining <= 0:
            if player == self._move_start_player:
                failed_tile = _step(player, self._move_action)
                recovery = self._recovery_for_failed_move(symbol_map, player, self._move_action)
                if recovery:
                    self._recovery_actions = recovery[1:]
                    print(f"DEBUG recovery start: failed_tile={failed_tile}, actions={recovery}, player={player}")
                    self._move_action = ACTION_NOOP
                    self._move_start_player = None
                    return recovery[0]
                if _in_bounds(failed_tile):
                    self._failed_tiles[failed_tile] = 80
                print(f"DEBUG move failed: start={self._move_start_player}, action={self._move_action}, failed_tile={failed_tile}")
            self._move_action = ACTION_NOOP
            self._move_start_player = None

        # Pixel movement needs the same direction repeated for roughly one tile.
        if self._move_ticks_remaining > 0 and self._move_action != ACTION_NOOP:
            if self._move_start_player is not None and player != self._move_start_player:
                print(f"DEBUG exec stop: tile_changed from={self._move_start_player} to={player}")
                self._move_ticks_remaining = 0
                self._move_action = ACTION_NOOP
                self._move_start_player = None
            else:
                self._move_ticks_remaining -= 1
                print(f"DEBUG exec: action={self._move_action}, remaining={self._move_ticks_remaining}, player_now={symbol_map.player}")
                return self._move_action

        if goal is None:
            return ACTION_NOOP

        planned_goal = goal
        if self._needs_adjacent_planning(symbol_map, goal):
            planned_goal = self._adjacent_goal(symbol_map, goal) or goal

        # DEBUG: inspect adjacent target selection.
        print(f"DEBUG adjacent: goal={goal}, planned_goal={planned_goal}, player={player}")
        # DEBUG: inspect direct neighbors of the current target.
        if goal is not None:
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nb = (goal[0] + dx, goal[1] + dy)
                if _in_bounds(nb):
                    print(f"  neighbor {nb} blocked={nb in set(symbol_map.blocked_tiles())}")

        actions = self._plan_actions(symbol_map, player, planned_goal)
        print(f"DEBUG plan_path: actions={actions[:10] if actions else 'EMPTY'}")
        if not actions:
            return ACTION_NOOP

        action = actions[0]
        self._move_action = action
        self._move_ticks_remaining = max(TILE_SIZE - 1, 0)
        self._move_start_player = player
        print(f"state={self.state}, player={player}, goal={goal}, inv={inventory}")

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

    def _interaction_action(
        self,
        symbol_map,
        inventory: dict,
        player: GridPos | None = None,
        *,
        allow_chests: bool = True,
    ) -> int | None:
        player = symbol_map.player if symbol_map.player is not None else player
        if player is None:
            return None

        targets = tuple(symbol_map.buttons) + tuple(symbol_map.switches)
        if allow_chests:
            targets = tuple(symbol_map.chests) + targets
        for target in targets:
            if _manhattan(player, target) <= 1:
                self._interaction_cooldown = 2
                print(f"DEBUG interact A: player={player}, inventory before={inventory}")
                return ACTION_A

        if _has_inventory_item(inventory, "sword", "has_sword") and symbol_map.monsters:
            monster = min(symbol_map.monsters, key=lambda pos: _manhattan(player, pos))
            if _manhattan(player, monster) <= 1:
                self._interaction_cooldown = 2
                print(f"DEBUG interact A: player={player}, inventory before={inventory}")
                return ACTION_A
        return None

    def _is_task3(self) -> bool:
        return "task_3" in self.task_id.lower() or "task3" in self.task_id.lower()

    def _apply_task3_key_search_state(self, symbol_map, inventory: dict) -> None:
        if _inventory_key_count(inventory) > 0:
            return
        if symbol_map.chests:
            self.state = TaskState.GET_KEY
            return
        if symbol_map.monsters and _has_inventory_item(inventory, "sword", "has_sword"):
            self.state = TaskState.KILL_GUARDIAN
            return
        if self.state == TaskState.GO_TO_EXIT:
            self.state = TaskState.GET_KEY

    def _task3_route_goal(self, symbol_map, inventory: dict, player: GridPos) -> GridPos | None:
        if not self._is_task3():
            return None
        key_count = _inventory_key_count(inventory)
        if key_count > 0:
            if not symbol_map.exits:
                return None
            return self._choose_room_exit(symbol_map, player)
        if symbol_map.chests:
            return min(symbol_map.chests, key=lambda pos: _manhattan(player, pos))
        if self.state == TaskState.KILL_GUARDIAN and symbol_map.monsters:
            return min(symbol_map.monsters, key=lambda pos: _manhattan(player, pos))
        if not symbol_map.exits:
            return None
        return self._choose_room_exit(symbol_map, player)

    def _choose_room_exit(self, symbol_map, player: GridPos) -> GridPos | None:
        exits = tuple(symbol_map.exits)
        if not exits:
            return None
        candidates = [exit_pos for exit_pos in exits if _boundary_side(exit_pos) != self._entry_side]
        if not candidates:
            candidates = list(exits)
        return min(candidates, key=lambda pos: _manhattan(player, pos))

    def _needs_adjacent_planning(self, symbol_map, goal: GridPos) -> bool:
        interactive_targets = (
            set(symbol_map.chests)
            | set(symbol_map.monsters)
            | set(symbol_map.buttons)
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
        if self.state == TaskState.GO_TO_EXIT:
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
        occupancy[start[1], start[0]] = 0
        occupancy[goal[1], goal[0]] = 0

        approach = _inside_exit_approach(goal)
        if approach is not None and _in_bounds(approach):
            occupancy[approach[1], approach[0]] = 0
            if start == approach:
                return [_action_between(start, goal)]
            path_to_approach = a_star(occupancy, start, approach)
            if path_to_approach:
                return path_to_actions([start, *path_to_approach, goal])

        path = a_star(occupancy, start, goal)
        return path_to_actions([start, *path]) if path else []

    def _recovery_for_failed_move(self, symbol_map, player: GridPos, action: int) -> list[int]:
        failed_tile = _step(player, action)
        if not _in_bounds(failed_tile):
            return []

        blocked = set(symbol_map.blocked_tiles()) | set(symbol_map.monsters)
        nudge_action = ACTION_NOOP
        if action in (ACTION_UP, ACTION_DOWN):
            left_side = (failed_tile[0] - 1, failed_tile[1])
            right_side = (failed_tile[0] + 1, failed_tile[1])
            left_blocked = (not _in_bounds(left_side)) or left_side in blocked
            right_blocked = (not _in_bounds(right_side)) or right_side in blocked
            if left_blocked and not right_blocked:
                nudge_action = ACTION_RIGHT
            elif right_blocked and not left_blocked:
                nudge_action = ACTION_LEFT
            else:
                nudge_action = ACTION_RIGHT if _can_step(player, ACTION_RIGHT, blocked) else ACTION_LEFT
        elif action in (ACTION_LEFT, ACTION_RIGHT):
            upper_side = (failed_tile[0], failed_tile[1] - 1)
            lower_side = (failed_tile[0], failed_tile[1] + 1)
            upper_blocked = (not _in_bounds(upper_side)) or upper_side in blocked
            lower_blocked = (not _in_bounds(lower_side)) or lower_side in blocked
            if upper_blocked and not lower_blocked:
                nudge_action = ACTION_DOWN
            elif lower_blocked and not upper_blocked:
                nudge_action = ACTION_UP
            else:
                nudge_action = ACTION_DOWN if _can_step(player, ACTION_DOWN, blocked) else ACTION_UP

        if nudge_action == ACTION_NOOP or not _can_step(player, nudge_action, blocked):
            return []
        nudge_ticks = max(4, TILE_SIZE // 3)
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
        if player in self._known_exits:
            return True
        if self._locked_exit is None:
            return False
        if player[1] == self._locked_exit[1] and player[1] in (0, GRID_HEIGHT - 1):
            return abs(player[0] - self._locked_exit[0]) <= 1
        if player[0] == self._locked_exit[0] and player[0] in (0, GRID_WIDTH - 1):
            return abs(player[1] - self._locked_exit[1]) <= 1
        return False

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
    """Return a length-7 action mask for debugging or formal checks."""

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
    total_reward = 0.0

    for step_idx in range(1, max_steps + 1):
        frame = _extract_frame(obs, env)
        # Debug only: final evaluation should pass inventory explicitly.
        inventory = _normalize_inventory(info.get("inventory", {})) if isinstance(info, dict) else {}
        action = agent.step(frame, inventory)
        obs, reward, terminated, truncated, info = env.step(action)
        agent.update_reward(float(reward))
        total_reward += float(reward)
        if terminated or truncated:
            print(
                f"episode finished: steps={step_idx}, reward={total_reward:.3f}, "
                f"terminated={terminated}, truncated={truncated}"
            )
            return

    print(f"episode reached max_steps: steps={max_steps}, reward={total_reward:.3f}")


def _extract_frame(obs: Any, env: Any | None = None) -> np.ndarray:
    if isinstance(obs, dict):
        if "frame" in obs or "obs" in obs:
            normalized = normalize_agent_observation(obs)
            return normalized.frame
        if env is not None:
            # Debug only: full/grid observations are not used by the policy.
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
