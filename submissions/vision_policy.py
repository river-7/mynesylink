from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable

from nesylink.core.constants import (
    ACTION_A,
    ACTION_DOWN,
    ACTION_LEFT,
    ACTION_NOOP,
    ACTION_RIGHT,
    ACTION_UP,
    TILE_SIZE,
)
from submissions.vision import GridPos, SymbolMap, VisionState, normalize_agent_observation


MOVE_ACTIONS = {
    (0, -1): ACTION_UP,
    (0, 1): ACTION_DOWN,
    (-1, 0): ACTION_LEFT,
    (1, 0): ACTION_RIGHT,
}
ACTION_TO_DELTA = {action: delta for delta, action in MOVE_ACTIONS.items()}


@dataclass
class GoalPlan:
    target: GridPos
    path: list[GridPos]


def manhattan(left: GridPos, right: GridPos) -> int:
    return abs(left[0] - right[0]) + abs(left[1] - right[1])


def neighbors(pos: GridPos) -> Iterable[GridPos]:
    x, y = pos
    yield (x, y - 1)
    yield (x, y + 1)
    yield (x - 1, y)
    yield (x + 1, y)


def action_between(current: GridPos, nxt: GridPos) -> int:
    delta = (nxt[0] - current[0], nxt[1] - current[1])
    return MOVE_ACTIONS.get(delta, ACTION_NOOP)


def outward_action(edge_tile: GridPos) -> int:
    x, y = edge_tile
    if y == 0:
        return ACTION_UP
    if y == 7:
        return ACTION_DOWN
    if x == 0:
        return ACTION_LEFT
    if x == 9:
        return ACTION_RIGHT
    return ACTION_NOOP


def bfs(
    symbol_map: SymbolMap,
    goals: Iterable[GridPos],
    *,
    avoid_danger: bool = True,
    extra_danger: Iterable[GridPos] = (),
) -> GoalPlan | None:
    if symbol_map.player is None:
        return None
    goal_set = set(goals)
    if not goal_set:
        return None

    start = symbol_map.player
    blocked = symbol_map.blocked_tiles()
    danger = symbol_map.danger_tiles() if avoid_danger else set()
    danger |= set(extra_danger)
    queue: deque[GridPos] = deque([start])
    parent: dict[GridPos, GridPos | None] = {start: None}

    while queue:
        current = queue.popleft()
        if current in goal_set:
            path: list[GridPos] = []
            cursor: GridPos | None = current
            while cursor is not None:
                path.append(cursor)
                cursor = parent[cursor]
            path.reverse()
            return GoalPlan(target=current, path=path)

        for nxt in neighbors(current):
            if nxt in parent:
                continue
            if not symbol_map.in_bounds(nxt):
                continue
            if nxt in blocked and nxt not in goal_set:
                continue
            if nxt in danger and nxt not in goal_set:
                continue
            parent[nxt] = current
            queue.append(nxt)
    return None


def adjacent_passable(symbol_map: SymbolMap, target: GridPos, *, avoid_danger: bool = True) -> set[GridPos]:
    blocked = symbol_map.blocked_tiles()
    danger = symbol_map.danger_tiles() if avoid_danger else set()
    result: set[GridPos] = set()
    for pos in neighbors(target):
        if not symbol_map.in_bounds(pos):
            continue
        if pos in blocked:
            continue
        if pos in danger:
            continue
        result.add(pos)
    return result


class Policy:
    """Minimal frame-only policy using A's vision layer.

    This is an integration harness, not the final group planner. It deliberately
    ignores `info` during `act`; task-specific phase memory comes from observed
    pixels and prior actions/rewards.
    """

    def __init__(self) -> None:
        self.vision = VisionState()
        self.task_id = ""
        self.phase = "start"
        self.last_reward = 0.0
        self.attack_cooldown = 0
        self.interaction_cooldown = 0
        self.exit_pushes = 0
        self.move_action = ACTION_NOOP
        self.move_ticks_remaining = 0
        self.known_exits: set[GridPos] = set()
        self.opened_chests: set[GridPos] = set()
        self.last_player_cell: GridPos | None = None
        self.same_cell_ticks = 0
        self.task3_has_key = False
        self.task4_has_key = False
        self.task4_has_sword = False
        self.task4_switch_presses = 0
        self.task4_monster_killed = False
        self.task5_button_pressed = False
        self.task5_west_done = False
        self.task5_has_key = False
        self.task5_east_done = False
        self.task5_start_gold_done = False

    def reset(self, seed: int | None = None, task_id: str | None = None) -> None:
        del seed
        self.vision.reset()
        self.task_id = task_id or ""
        self.phase = "start"
        self.last_reward = 0.0
        self.attack_cooldown = 0
        self.interaction_cooldown = 0
        self.exit_pushes = 0
        self.move_action = ACTION_NOOP
        self.move_ticks_remaining = 0
        self.known_exits = set()
        self.opened_chests = set()
        self.last_player_cell = None
        self.same_cell_ticks = 0
        self.task3_has_key = False
        self.task4_has_key = False
        self.task4_has_sword = False
        self.task4_switch_presses = 0
        self.task4_monster_killed = False
        self.task5_button_pressed = False
        self.task5_west_done = False
        self.task5_has_key = False
        self.task5_east_done = False
        self.task5_start_gold_done = False

    def act(self, obs, info=None) -> int:
        del info
        normalized = normalize_agent_observation(obs, reward=self.last_reward)
        symbol_map = self.vision.observe(normalized.frame, reward=normalized.reward)
        if symbol_map.player is None:
            return ACTION_NOOP
        if self.last_player_cell is not None and manhattan(symbol_map.player, self.last_player_cell) > 3:
            self.move_ticks_remaining = 0
            self.move_action = ACTION_NOOP
            self.exit_pushes = 0
        if symbol_map.player == self.last_player_cell:
            self.same_cell_ticks += 1
        else:
            self.same_cell_ticks = 0
        self.last_player_cell = symbol_map.player
        self.known_exits.update(symbol_map.exits)
        if self.attack_cooldown > 0:
            self.attack_cooldown -= 1

        if self._is_exit_phase() and self._player_is_on_exit(symbol_map):
            self.move_ticks_remaining = 0
            self.move_action = ACTION_NOOP
            return self._go_to_exit(symbol_map)

        interrupt_action = self._urgent_interaction(symbol_map)
        if interrupt_action is not None:
            self.move_ticks_remaining = 0
            self.move_action = ACTION_NOOP
            return interrupt_action

        if self.move_ticks_remaining > 0:
            if self.same_cell_ticks < 12:
                self.move_ticks_remaining -= 1
                return self.move_action
            self.move_ticks_remaining = 0
            self.move_action = ACTION_NOOP

        if self.task_id.endswith("task_5"):
            return self._act_task5(symbol_map)
        if self.task_id.endswith("task_4"):
            return self._act_task4(symbol_map)
        if self.task_id.endswith("task_3"):
            return self._act_task3(symbol_map)
        if self.task_id.endswith("task_2"):
            return self._act_task2(symbol_map)
        return self._act_key_then_exit(symbol_map)

    def _is_exit_phase(self) -> bool:
        return self.phase == "seek_exit" or self.phase.startswith("task3_exit_")

    def _urgent_interaction(self, symbol_map: SymbolMap) -> int | None:
        if symbol_map.player is None:
            return None
        task5_west_fight = (
            self.task_id.endswith("task_5")
            and symbol_map.monsters
            and (
                (self._is_task5_start_room(symbol_map) and self.task5_east_done)
                or self.phase == "task5_west"
            )
        )
        if (self.task_id.endswith("task_2") or task5_west_fight) and symbol_map.monsters:
            monster = min(symbol_map.monsters, key=lambda pos: manhattan(symbol_map.player, pos))
            if manhattan(symbol_map.player, monster) <= 1:
                if self.attack_cooldown <= 0:
                    self.attack_cooldown = 5
                    return ACTION_A
                self.attack_cooldown -= 1
                return self._face_target(symbol_map.player, monster)
        visible_chests = self._unopened_chests(symbol_map)
        if self.phase == "seek_chest" and visible_chests:
            chest = min(visible_chests, key=lambda pos: manhattan(symbol_map.player, pos))
            if manhattan(symbol_map.player, chest) <= 1:
                self.opened_chests.add(chest)
                if self.task_id.endswith("task_3"):
                    self.task3_has_key = True
                    self.phase = "task3_exit_east"
                else:
                    self.phase = "seek_exit"
                self.interaction_cooldown = 8
                return ACTION_A
        return None

    def _act_key_then_exit(self, symbol_map: SymbolMap) -> int:
        if self.phase == "start":
            self.phase = "seek_chest" if self._unopened_chests(symbol_map) else "seek_exit"

        if self.phase == "seek_chest":
            visible_chests = self._unopened_chests(symbol_map)
            if not visible_chests:
                self.phase = "seek_exit"
            else:
                chest = min(visible_chests, key=lambda pos: manhattan(symbol_map.player, pos))  # type: ignore[arg-type]
                if manhattan(symbol_map.player, chest) <= 1:  # type: ignore[arg-type]
                    self.opened_chests.add(chest)
                    self.phase = "seek_exit"
                    self.interaction_cooldown = 8
                    return ACTION_A
                return self._move_toward_any(symbol_map, adjacent_passable(symbol_map, chest))

        if self.interaction_cooldown > 0:
            self.interaction_cooldown -= 1
            return ACTION_NOOP

        return self._go_to_exit(symbol_map)

    def _act_task2(self, symbol_map: SymbolMap) -> int:
        if symbol_map.monsters:
            monster = min(symbol_map.monsters, key=lambda pos: manhattan(symbol_map.player, pos))  # type: ignore[arg-type]
            distance = manhattan(symbol_map.player, monster)  # type: ignore[arg-type]
            if distance <= 1:
                if self.attack_cooldown <= 0:
                    self.attack_cooldown = 5
                    return ACTION_A
                self.attack_cooldown -= 1
                return self._face_target(symbol_map.player, monster)  # type: ignore[arg-type]
            return self._move_toward_any(
                symbol_map,
                adjacent_passable(symbol_map, monster, avoid_danger=False),
                avoid_danger=False,
            )

        self.attack_cooldown = 0
        return self._act_key_then_exit(symbol_map)

    def _act_task3(self, symbol_map: SymbolMap) -> int:
        visible_chests = self._unopened_chests(symbol_map)
        if visible_chests:
            self.phase = "seek_chest"
            chest = min(visible_chests, key=lambda pos: manhattan(symbol_map.player, pos))  # type: ignore[arg-type]
            if manhattan(symbol_map.player, chest) <= 1:  # type: ignore[arg-type]
                self.opened_chests.add(chest)
                self.task3_has_key = True
                self.phase = "task3_exit_east"
                self.interaction_cooldown = 8
                return ACTION_A
            return self._move_toward_any(symbol_map, adjacent_passable(symbol_map, chest))

        if symbol_map.monsters:
            return self._act_task2(symbol_map)

        if self.interaction_cooldown > 0:
            self.interaction_cooldown -= 1
            return ACTION_NOOP

        if symbol_map.npcs:
            self.phase = "task3_exit_east" if self.task3_has_key else "task3_exit_west"
        elif self.task3_has_key:
            self.phase = "task3_exit_east"
        else:
            self.phase = "task3_exit_west"

        direction = "east" if self.phase == "task3_exit_east" else "west"
        return self._go_to_direction_exit(symbol_map, direction)

    def _act_task4(self, symbol_map: SymbolMap) -> int:
        visible_chests = self._unopened_chests(symbol_map)
        if visible_chests:
            chest = min(visible_chests, key=lambda pos: manhattan(symbol_map.player, pos))  # type: ignore[arg-type]
            if manhattan(symbol_map.player, chest) <= 1:  # type: ignore[arg-type]
                self.opened_chests.add(chest)
                if not self.task4_has_key:
                    self.task4_has_key = True
                    self.phase = "task4_return_switch_for_east"
                elif not self.task4_has_sword:
                    self.task4_has_sword = True
                    self.phase = "task4_return_switch_for_south"
                else:
                    self.phase = "task4_done"
                self.interaction_cooldown = 8
                return ACTION_A
            diagonal_step = self._diagonal_to_cardinal_step(symbol_map, chest)
            if diagonal_step is not None:
                return diagonal_step
            return self._move_toward_any(symbol_map, adjacent_passable(symbol_map, chest))

        if symbol_map.monsters:
            self.phase = "task4_fight"
            monster = min(symbol_map.monsters, key=lambda pos: manhattan(symbol_map.player, pos))  # type: ignore[arg-type]
            distance = manhattan(symbol_map.player, monster)  # type: ignore[arg-type]
            if distance <= 1:
                if self.attack_cooldown <= 0:
                    self.attack_cooldown = 5
                    self.task4_monster_killed = True
                    return ACTION_A
                self.attack_cooldown -= 1
                return self._face_target(symbol_map.player, monster)  # type: ignore[arg-type]
            return self._move_toward_any(
                symbol_map,
                adjacent_passable(symbol_map, monster, avoid_danger=False),
                avoid_danger=False,
            )

        if self.interaction_cooldown > 0:
            self.interaction_cooldown -= 1
            return ACTION_NOOP

        if symbol_map.switches and self._task4_needs_switch_press():
            switch = min(symbol_map.switches, key=lambda pos: manhattan(symbol_map.player, pos))  # type: ignore[arg-type]
            if manhattan(symbol_map.player, switch) <= 1:  # type: ignore[arg-type]
                self.task4_switch_presses += 1
                self.interaction_cooldown = 8
                return ACTION_A
            return self._move_toward_any(symbol_map, adjacent_passable(symbol_map, switch))

        if symbol_map.switches:
            return self._go_to_direction_exit(symbol_map, "east")

        if not symbol_map.bridges and not symbol_map.gaps and symbol_map.exits:
            return self._go_to_current_room_exit(symbol_map)

        direction = self._task4_center_direction()
        return self._go_to_direction_exit(symbol_map, direction)

    def _task4_needs_switch_press(self) -> bool:
        if not self.task4_has_sword:
            return self.task4_has_key and self.task4_switch_presses < 1
        if not self.task4_monster_killed:
            return self.task4_switch_presses < 2
        return False

    def _task4_center_direction(self) -> str:
        if not self.task4_has_key:
            return "north"
        if not self.task4_has_sword:
            return "east" if self.task4_switch_presses >= 1 else "west"
        if not self.task4_monster_killed:
            return "south" if self.task4_switch_presses >= 2 else "west"
        return "north"

    def _act_task5(self, symbol_map: SymbolMap) -> int:
        in_start_room = self._is_task5_start_room(symbol_map)
        if symbol_map.monsters:
            adjacent_monsters = [
                monster
                for monster in symbol_map.monsters
                if symbol_map.player is not None and manhattan(symbol_map.player, monster) <= 1
            ]
            if adjacent_monsters:
                monster = min(adjacent_monsters, key=lambda pos: manhattan(symbol_map.player, pos))  # type: ignore[arg-type]
                if self.attack_cooldown <= 0:
                    self.attack_cooldown = 5
                    return ACTION_A
                self.attack_cooldown -= 1
                return self._face_target(symbol_map.player, monster)  # type: ignore[arg-type]
            if (in_start_room and self.task5_east_done) or (self.phase == "task5_west"):
                return self._engage_nearest_monster(symbol_map)

        if self.interaction_cooldown > 0:
            self.interaction_cooldown -= 1
            return ACTION_NOOP

        if in_start_room:
            if symbol_map.buttons and not self.task5_button_pressed:
                button = min(symbol_map.buttons, key=lambda pos: manhattan(symbol_map.player, pos))  # type: ignore[arg-type]
                if symbol_map.player == button:
                    self.task5_button_pressed = True
                    self.interaction_cooldown = 8
                    return ACTION_NOOP
                return self._move_toward_any(symbol_map, {button})
            if not self.task5_has_key:
                self.phase = "task5_south"
                return self._go_to_direction_exit(symbol_map, "south")
            if not self.task5_east_done:
                self.phase = "task5_east"
                return self._go_to_direction_exit(symbol_map, "east")
            if not self.task5_start_gold_done:
                visible_chests = self._unopened_chests(symbol_map)
                if visible_chests:
                    self.phase = "task5_start_gold"
                    chest = min(visible_chests, key=lambda pos: manhattan(symbol_map.player, pos))  # type: ignore[arg-type]
                    if manhattan(symbol_map.player, chest) <= 1:  # type: ignore[arg-type]
                        self.opened_chests.add(chest)
                        self.task5_start_gold_done = True
                        self.interaction_cooldown = 8
                        return ACTION_A
                    return self._move_toward_any(symbol_map, adjacent_passable(symbol_map, chest))
                self.task5_start_gold_done = True
            if not self.task5_west_done:
                self.phase = "task5_west"
                return self._go_to_direction_exit(symbol_map, "west")
            self.phase = "task5_cleanup"

        visible_chests = self._unopened_chests(symbol_map)
        if visible_chests:
            chest = min(visible_chests, key=lambda pos: manhattan(symbol_map.player, pos))  # type: ignore[arg-type]
            if manhattan(symbol_map.player, chest) <= 1:  # type: ignore[arg-type]
                self.opened_chests.add(chest)
                if self.phase == "task5_west":
                    self.task5_west_done = True
                elif self.phase == "task5_south":
                    self.task5_has_key = True
                elif self.phase == "task5_east":
                    self.task5_east_done = True
                self.interaction_cooldown = 8
                return ACTION_A
            return self._move_toward_any(
                symbol_map,
                adjacent_passable(symbol_map, chest),
                extra_danger=self._monster_zones(symbol_map) if self.phase == "task5_west" else (),
            )

        return self._go_to_current_room_exit(symbol_map)

    def _engage_nearest_monster(self, symbol_map: SymbolMap) -> int:
        if symbol_map.player is None or not symbol_map.monsters:
            return ACTION_NOOP
        monster = min(symbol_map.monsters, key=lambda pos: manhattan(symbol_map.player, pos))
        return self._move_toward_any(symbol_map, adjacent_passable(symbol_map, monster), avoid_danger=False)

    def _monster_zones(self, symbol_map: SymbolMap) -> set[GridPos]:
        zones = set(symbol_map.monsters)
        for monster in symbol_map.monsters:
            zones.update(pos for pos in neighbors(monster) if symbol_map.in_bounds(pos))
        return zones

    def _is_task5_start_room(self, symbol_map: SymbolMap) -> bool:
        return len(symbol_map.exits) >= 5

    def _go_to_exit(self, symbol_map: SymbolMap) -> int:
        if self._player_is_on_exit(symbol_map):
            if self.exit_pushes < 24:
                self.exit_pushes += 1
                return outward_action(symbol_map.player)
            return ACTION_NOOP
        self.exit_pushes = 0
        return self._move_toward_any(symbol_map, self.known_exits or set(symbol_map.exits))

    def _go_to_direction_exit(
        self,
        symbol_map: SymbolMap,
        direction: str,
        *,
        avoid_danger: bool = True,
        extra_danger: Iterable[GridPos] = (),
    ) -> int:
        exits = self._direction_exits(symbol_map, direction)
        if symbol_map.player in exits:
            if self.exit_pushes < 24:
                self.exit_pushes += 1
                return outward_action(symbol_map.player)
            return ACTION_NOOP
        self.exit_pushes = 0
        return self._move_toward_any(symbol_map, exits, avoid_danger=avoid_danger, extra_danger=extra_danger)

    def _go_to_current_room_exit(self, symbol_map: SymbolMap) -> int:
        exits = set(symbol_map.exits)
        if not exits and symbol_map.player is not None:
            x, y = symbol_map.player
            if x in (0, 9) or y in (0, 7):
                if self.exit_pushes < 24:
                    self.exit_pushes += 1
                    return outward_action(symbol_map.player)
                return ACTION_NOOP
        if symbol_map.player in exits or self._player_is_on_visible_exit_segment(symbol_map, exits):
            if self.exit_pushes < 24:
                self.exit_pushes += 1
                return outward_action(symbol_map.player)
            return ACTION_NOOP
        self.exit_pushes = 0
        return self._move_toward_any(symbol_map, exits)

    def _player_is_on_visible_exit_segment(self, symbol_map: SymbolMap, exits: set[GridPos]) -> bool:
        if symbol_map.player is None or not exits:
            return False
        x, y = symbol_map.player
        if x not in (0, 9) and y not in (0, 7):
            return False
        exit_xs = [pos[0] for pos in exits]
        exit_ys = [pos[1] for pos in exits]
        if y == 0 and any(exit_y == 0 for exit_y in exit_ys):
            top_xs = [pos[0] for pos in exits if pos[1] == 0]
            margin = 1 if len(top_xs) == 1 else 0
            return min(top_xs) - margin <= x <= max(top_xs) + margin
        if y == 7 and any(exit_y == 7 for exit_y in exit_ys):
            bottom_xs = [pos[0] for pos in exits if pos[1] == 7]
            margin = 1 if len(bottom_xs) == 1 else 0
            return min(bottom_xs) - margin <= x <= max(bottom_xs) + margin
        if x == 0 and any(exit_x == 0 for exit_x in exit_xs):
            left_ys = [pos[1] for pos in exits if pos[0] == 0]
            margin = 1 if len(left_ys) == 1 else 0
            return min(left_ys) - margin <= y <= max(left_ys) + margin
        if x == 9 and any(exit_x == 9 for exit_x in exit_xs):
            right_ys = [pos[1] for pos in exits if pos[0] == 9]
            margin = 1 if len(right_ys) == 1 else 0
            return min(right_ys) - margin <= y <= max(right_ys) + margin
        return False

    def _diagonal_to_cardinal_step(self, symbol_map: SymbolMap, target: GridPos) -> int | None:
        if symbol_map.player is None:
            return None
        player = symbol_map.player
        dx = target[0] - player[0]
        dy = target[1] - player[1]
        if abs(dx) != 1 or abs(dy) != 1:
            return None
        candidates = ((player[0] + dx, player[1]), (player[0], player[1] + dy))
        for candidate in candidates:
            if symbol_map.in_bounds(candidate) and not symbol_map.is_blocked(candidate):
                action = action_between(player, candidate)
                if action != ACTION_NOOP:
                    self.move_action = action
                    self.move_ticks_remaining = TILE_SIZE - 1
                    return action
        return None

    def _unopened_chests(self, symbol_map: SymbolMap) -> tuple[GridPos, ...]:
        return tuple(chest for chest in symbol_map.chests if chest not in self.opened_chests)

    def _direction_exits(self, symbol_map: SymbolMap, direction: str) -> set[GridPos]:
        exits = set(symbol_map.exits) | self.known_exits
        if direction == "west":
            return {pos for pos in exits if pos[0] == 0}
        if direction == "east":
            return {pos for pos in exits if pos[0] == 9}
        if direction == "north":
            return {pos for pos in exits if pos[1] == 0}
        if direction == "south":
            return {pos for pos in exits if pos[1] == 7}
        return exits

    def _player_is_on_exit(self, symbol_map: SymbolMap) -> bool:
        if symbol_map.player is None:
            return False
        x, y = symbol_map.player
        exits = self.known_exits or set(symbol_map.exits)
        if symbol_map.player in exits:
            return True
        if not exits:
            return False
        exit_xs = [pos[0] for pos in exits]
        exit_ys = [pos[1] for pos in exits]
        if y == 0 and any(exit_y == 0 for exit_y in exit_ys):
            return min(exit_xs) <= x <= max(exit_xs)
        if y == 7 and any(exit_y == 7 for exit_y in exit_ys):
            return min(exit_xs) <= x <= max(exit_xs)
        if x == 0 and any(exit_x == 0 for exit_x in exit_xs):
            return min(exit_ys) <= y <= max(exit_ys)
        if x == 9 and any(exit_x == 9 for exit_x in exit_xs):
            return min(exit_ys) <= y <= max(exit_ys)
        return False

    def _move_toward_any(
        self,
        symbol_map: SymbolMap,
        goals: Iterable[GridPos],
        *,
        avoid_danger: bool = True,
        extra_danger: Iterable[GridPos] = (),
    ) -> int:
        plan = bfs(symbol_map, goals, avoid_danger=avoid_danger, extra_danger=extra_danger)
        if plan is None or len(plan.path) < 2:
            return ACTION_NOOP
        action = action_between(plan.path[0], plan.path[1])
        if action != ACTION_NOOP:
            self.move_action = action
            self.move_ticks_remaining = TILE_SIZE - 1
        return action

    def _face_target(self, current: GridPos, target: GridPos) -> int:
        dx = target[0] - current[0]
        dy = target[1] - current[1]
        if abs(dx) > abs(dy):
            return ACTION_RIGHT if dx > 0 else ACTION_LEFT
        if dy != 0:
            return ACTION_DOWN if dy > 0 else ACTION_UP
        return ACTION_NOOP


def make_policy() -> Policy:
    return Policy()
