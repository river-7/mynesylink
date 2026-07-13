from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from functools import lru_cache
from typing import Any, Iterable

import numpy as np

from nesylink.core.constants import GRID_HEIGHT, GRID_WIDTH, TILE_SIZE
from nesylink.core.constants import (
    COLOR_EXIT_CONDITIONAL,
    COLOR_EXIT_LOCKED,
    COLOR_EXIT_NORMAL,
    COLOR_MONSTER_AMBUSHER,
    COLOR_MONSTER_CHASER,
    COLOR_MONSTER_PATROLLER,
    COLOR_NPC,
)
from nesylink.core.rendering import sprites as sprite_colors


class Cell(IntEnum):
    EMPTY = 0
    WALL = 1
    PLAYER = 2
    MONSTER = 3
    CHEST = 4
    EXIT = 5
    TRAP = 6
    BUTTON = 7
    NPC = 8
    GAP = 9
    BRIDGE = 10
    SWITCH = 11


GridPos = tuple[int, int]
Color = tuple[int, int, int]
ColorVariant = str


@dataclass(frozen=True)
class SymbolMap:
    """Symbolic map extracted only from an RGB frame."""

    grid: np.ndarray
    player: GridPos | None
    monsters: tuple[GridPos, ...]
    chests: tuple[GridPos, ...]
    exits: tuple[GridPos, ...]
    walls: tuple[GridPos, ...]
    traps: tuple[GridPos, ...]
    buttons: tuple[GridPos, ...]
    switches: tuple[GridPos, ...]
    npcs: tuple[GridPos, ...]
    gaps: tuple[GridPos, ...]
    bridges: tuple[GridPos, ...]

    def cell_at(self, pos: GridPos) -> Cell:
        x, y = pos
        return Cell(int(self.grid[y, x]))

    def in_bounds(self, pos: GridPos) -> bool:
        x, y = pos
        return 0 <= x < GRID_WIDTH and 0 <= y < GRID_HEIGHT

    def blocked_tiles(self) -> set[GridPos]:
        return set(self.walls) | set(self.chests) | set(self.npcs) | set(self.gaps)

    def danger_tiles(self) -> set[GridPos]:
        return set(self.traps) | set(self.monsters)

    def is_blocked(self, pos: GridPos) -> bool:
        return (not self.in_bounds(pos)) or pos in self.blocked_tiles()

    def is_dangerous(self, pos: GridPos) -> bool:
        return pos in self.danger_tiles()

    def passable_tiles(self, *, avoid_danger: bool = True) -> set[GridPos]:
        blocked = self.blocked_tiles()
        if avoid_danger:
            blocked |= self.danger_tiles()
        return {
            (x, y)
            for y in range(GRID_HEIGHT)
            for x in range(GRID_WIDTH)
            if (x, y) not in blocked
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "grid": self.grid.tolist(),
            "player": self.player,
            "monsters": self.monsters,
            "chests": self.chests,
            "exits": self.exits,
            "walls": self.walls,
            "traps": self.traps,
            "buttons": self.buttons,
            "switches": self.switches,
            "npcs": self.npcs,
            "gaps": self.gaps,
            "bridges": self.bridges,
        }

    def as_ascii(self) -> str:
        chars = {
            Cell.EMPTY: ".",
            Cell.WALL: "#",
            Cell.PLAYER: "P",
            Cell.MONSTER: "M",
            Cell.CHEST: "C",
            Cell.EXIT: "E",
            Cell.TRAP: "T",
            Cell.BUTTON: "B",
            Cell.NPC: "N",
            Cell.GAP: "G",
            Cell.BRIDGE: "=",
            Cell.SWITCH: "S",
        }
        return "\n".join(
            "".join(chars.get(Cell(int(value)), "?") for value in row)
            for row in self.grid
        )


@dataclass(frozen=True)
class AgentObservation:
    """Allowed final-policy inputs after lightweight normalization."""

    frame: np.ndarray
    inventory: tuple[int, ...] = ()
    reward: float = 0.0


_PLAYER_GREEN = (sprite_colors.PLAYER_TUNIC, sprite_colors.PLAYER_TUNIC_LIGHT)
_NPC_BODY = (sprite_colors.PLAYER_FACE,)
_MONSTER_BODY = (
    (238, 126, 28),
    (255, 180, 48),
    (200, 78, 16),
)
_CHEST_WOOD = (sprite_colors.CHEST_WOOD,)
_WALL = (
    sprite_colors.WALL_LIGHT,
    sprite_colors.WALL_MID,
    sprite_colors.WALL_DARK,
    sprite_colors.WALL_EDGE,
)
_EXIT = (
    sprite_colors.EXIT_GLOW,
    sprite_colors.DOOR_WOOD,
    sprite_colors.CONDITIONAL_GLYPH,
)
_TRAP = (
    sprite_colors.SPIKE_METAL,
    sprite_colors.SPIKE_SHADE,
    sprite_colors.SPIKE_HIGHLIGHT,
)
_BUTTON = (sprite_colors.BUTTON_UP, sprite_colors.BUTTON_DOWN, (86, 146, 104))
_SWITCH = (sprite_colors.SWITCH_BODY, sprite_colors.SWITCH_DOWN)
_GAP = (sprite_colors.GAP_DARK, sprite_colors.GAP_MID, (0, 0, 0))
_BRIDGE = (sprite_colors.BRIDGE_WOOD, sprite_colors.BRIDGE_EDGE)

_COLOR_VARIANTS: tuple[ColorVariant, ...] = (
    "default",
    "grayscale",
    "dark",
    "bright",
    "high_contrast",
    "inverted",
)
_PALETTE_PROBES = tuple({
    sprite_colors.FLOOR_LIGHT,
    sprite_colors.FLOOR_DARK,
    sprite_colors.FLOOR_DARKER,
    sprite_colors.OUTLINE,
    sprite_colors.HIGHLIGHT,
    *_PLAYER_GREEN,
    *_NPC_BODY,
    *_MONSTER_BODY,
    *_CHEST_WOOD,
    *_WALL,
    *_EXIT,
    *_TRAP,
    *_BUTTON,
    *_SWITCH,
    *_GAP,
    *_BRIDGE,
})


class VisionState:
    """Frame-to-symbol detector with optional static-map memory."""

    def __init__(self) -> None:
        self.static_grid = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=np.uint8)
        self._initialized = False
        self._last_player: GridPos | None = None
        self._last_monsters: tuple[GridPos, ...] = ()
        self._last_reward = 0.0

    def reset(self) -> None:
        self.static_grid[:, :] = Cell.EMPTY
        self._initialized = False
        self._last_player = None
        self._last_monsters = ()
        self._last_reward = 0.0

    def observe(self, frame: np.ndarray, *, reward: float | None = None) -> SymbolMap:
        grid = _detect_grid(frame)

        static_grid = grid.copy()
        dynamic_mask = (static_grid == Cell.PLAYER) | (static_grid == Cell.MONSTER)
        static_grid[dynamic_mask] = Cell.EMPTY

        if not self._initialized:
            self.static_grid = static_grid
            self._initialized = True
        else:
            visible_static = ~dynamic_mask
            self.static_grid[visible_static] = static_grid[visible_static]
            grid = self._merge_dynamic_with_memory(self.static_grid, grid)

        symbol_map = _to_symbol_map(grid)
        self._last_player = symbol_map.player
        self._last_monsters = symbol_map.monsters
        if reward is not None:
            self._last_reward = float(reward)
        return symbol_map

    def _merge_dynamic_with_memory(self, static_grid: np.ndarray, current_grid: np.ndarray) -> np.ndarray:
        merged = static_grid.copy()
        player = _choose_nearest(
            _positions(current_grid, Cell.PLAYER),
            self._last_player,
            max_distance=1,
        )
        if player is not None:
            merged[player[1], player[0]] = int(Cell.PLAYER)

        current_monsters = _positions(current_grid, Cell.MONSTER)
        monsters = _match_positions(current_monsters, self._last_monsters, max_distance=1)
        for monster in monsters:
            if player is not None and monster == player:
                continue
            merged[monster[1], monster[0]] = int(Cell.MONSTER)
        return merged


def normalize_agent_observation(obs: Any, reward: float = 0.0, inventory: Iterable[int] = ()) -> AgentObservation:
    """Normalize allowed policy inputs without reading hidden environment state.

    The public evaluator currently passes raw pixel arrays. This also accepts a
    dict containing `frame`/`obs` plus optional inventory fields, which matches
    the wording in the assignment README.
    """

    if isinstance(obs, dict):
        if "frame" in obs:
            frame = obs["frame"]
        elif "obs" in obs:
            frame = obs["obs"]
        else:
            raise KeyError("dict observation must contain 'frame' or 'obs'")
        inv = obs.get("inventory_ids", obs.get("inventory", inventory))
    else:
        frame = obs
        inv = inventory
    if inv is None:
        inventory_values = ()
    else:
        inventory_values = tuple(int(value) for value in np.asarray(inv, dtype=np.int64).reshape(-1))
    return AgentObservation(
        frame=_normalize_frame(frame),
        inventory=inventory_values,
        reward=float(reward),
    )


def detect(frame: np.ndarray) -> SymbolMap:
    """Detect a SymbolMap from one RGB frame.

    The function uses only pixels. It does not read `info`, map JSON, grid
    observations, or any runtime object coordinates.
    """

    return _to_symbol_map(_detect_grid(frame))


def _detect_grid(frame: np.ndarray) -> np.ndarray:
    frame = _normalize_frame(frame)
    color_variant = _detect_color_variant(frame)
    restrict_dynamic_tiles = color_variant in {"grayscale", "high_contrast"}
    grid = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=np.uint8)
    for y in range(GRID_HEIGHT):
        for x in range(GRID_WIDTH):
            top = y * TILE_SIZE
            left = x * TILE_SIZE
            tile = frame[top : top + TILE_SIZE, left : left + TILE_SIZE]
            grid[y, x] = int(_classify_static_tile(tile, x, y, color_variant=color_variant))

    _place_single_dynamic_tile(
        grid,
        frame,
        _variant_colors(_PLAYER_GREEN, color_variant),
        Cell.PLAYER,
        min_pixels=16,
        restrict_to_traversable=restrict_dynamic_tiles,
    )
    _place_dynamic_tile_groups(
        grid,
        frame,
        _variant_colors(_MONSTER_BODY, color_variant),
        Cell.MONSTER,
        min_pixels=12,
        restrict_to_traversable=restrict_dynamic_tiles,
    )
    return grid


def _normalize_frame(frame: np.ndarray) -> np.ndarray:
    array = np.asarray(frame)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"expected RGB frame with shape (H, W, 3), got {array.shape}")
    min_height = GRID_HEIGHT * TILE_SIZE
    min_width = GRID_WIDTH * TILE_SIZE
    if array.shape[0] < min_height or array.shape[1] < min_width:
        raise ValueError(f"frame is too small for the map area: {array.shape}")
    return array[:min_height, :min_width, :3]


def _classify_static_tile(
    tile: np.ndarray,
    x: int,
    y: int,
    *,
    color_variant: ColorVariant,
) -> Cell:
    if color_variant in {"grayscale", "high_contrast"}:
        return _classify_tile_by_template(tile, x, y, color_variant)

    player_green = _count_colors(tile, _variant_colors(_PLAYER_GREEN, color_variant))
    npc_body = _count_colors(tile, _variant_colors(_NPC_BODY, color_variant))
    chest_wood = _count_colors(tile, _variant_colors(_CHEST_WOOD, color_variant))
    wall = _count_colors(tile, _variant_colors(_WALL, color_variant))
    exit_pixels = _count_colors(tile, _variant_colors(_EXIT, color_variant))
    trap = _count_colors(tile, _variant_colors(_TRAP, color_variant))
    button = _count_colors(tile, _variant_colors(_BUTTON, color_variant))
    switch = _count_colors(tile, _variant_colors(_SWITCH, color_variant))
    gap = _count_colors(tile, _variant_colors(_GAP, color_variant))
    bridge = _count_colors(tile, _variant_colors(_BRIDGE, color_variant))
    edge_tile = x in (0, GRID_WIDTH - 1) or y in (0, GRID_HEIGHT - 1)

    if npc_body >= 25 and player_green < 10 and chest_wood < 10:
        return Cell.NPC
    if chest_wood >= 25:
        return Cell.CHEST
    if edge_tile and exit_pixels >= 18 and wall < 170:
        return Cell.EXIT
    if button >= 15:
        return Cell.BUTTON
    if switch >= 20 and chest_wood < 10:
        return Cell.SWITCH
    if bridge >= 60:
        return Cell.BRIDGE
    if trap >= 18:
        return Cell.TRAP
    if gap >= 100:
        return Cell.GAP
    if wall >= 120:
        return Cell.WALL
    return Cell.EMPTY


def _tile_color_counts(frame: np.ndarray, colors: Iterable[Color]) -> np.ndarray:
    counts = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=np.int32)
    for y in range(GRID_HEIGHT):
        for x in range(GRID_WIDTH):
            top = y * TILE_SIZE
            left = x * TILE_SIZE
            tile = frame[top : top + TILE_SIZE, left : left + TILE_SIZE]
            counts[y, x] = _count_colors(tile, colors)
    return counts


def _place_single_dynamic_tile(
    grid: np.ndarray,
    frame: np.ndarray,
    colors: Iterable[Color],
    cell: Cell,
    *,
    min_pixels: int,
    restrict_to_traversable: bool,
) -> None:
    counts = _tile_color_counts(frame, colors)
    if restrict_to_traversable:
        counts[~_dynamic_candidate_mask(grid, cell)] = 0
    best = int(counts.max())
    if best < min_pixels:
        return
    y, x = np.argwhere(counts == best)[0]
    grid[grid == int(cell)] = int(Cell.EMPTY)
    grid[int(y), int(x)] = int(cell)


def _place_dynamic_tile_groups(
    grid: np.ndarray,
    frame: np.ndarray,
    colors: Iterable[Color],
    cell: Cell,
    *,
    min_pixels: int,
    restrict_to_traversable: bool,
) -> None:
    counts = _tile_color_counts(frame, colors)
    if restrict_to_traversable:
        eligible = _dynamic_candidate_mask(grid, cell)
    else:
        eligible = np.ones_like(grid, dtype=bool)
    active = (counts >= min_pixels) & eligible
    seen = np.zeros_like(active, dtype=bool)
    grid[grid == int(cell)] = int(Cell.EMPTY)
    for start_y, start_x in np.argwhere(active):
        if seen[start_y, start_x]:
            continue
        stack = [(int(start_y), int(start_x))]
        seen[start_y, start_x] = True
        group: list[tuple[int, int]] = []
        while stack:
            y, x = stack.pop()
            group.append((y, x))
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny, nx = y + dy, x + dx
                if 0 <= ny < GRID_HEIGHT and 0 <= nx < GRID_WIDTH and active[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    stack.append((ny, nx))
        best_y, best_x = max(group, key=lambda pos: counts[pos[0], pos[1]])
        grid[best_y, best_x] = int(cell)


def _dynamic_candidate_mask(grid: np.ndarray, cell: Cell) -> np.ndarray:
    """Keep lossy-color matching away from solid objects, not traversable tiles.

    Players and monsters can visibly overlap exits, bridges, traps, buttons and
    switches while moving. Treating every non-empty tile as ineligible shifts
    them into a neighbouring tile and breaks room-transition tracking.
    """

    allowed = {int(Cell.EMPTY), int(cell)}
    if cell == Cell.PLAYER:
        allowed.update(
            {
                int(Cell.EXIT),
                int(Cell.TRAP),
                int(Cell.BUTTON),
                int(Cell.BRIDGE),
                int(Cell.SWITCH),
                int(Cell.GAP),
            }
        )
    return np.isin(grid, tuple(allowed))


def _count_colors(tile: np.ndarray, colors: Iterable[Color]) -> int:
    count = 0
    for color in colors:
        count += int(np.all(tile == color, axis=2).sum())
    return count


def _detect_color_variant(frame: np.ndarray) -> ColorVariant:
    """Infer the evaluator's deterministic color transform from pixels alone."""

    if np.all((frame == 0) | (frame == 255)):
        return "high_contrast"
    if np.array_equal(frame[:, :, 0], frame[:, :, 1]) and np.array_equal(frame[:, :, 1], frame[:, :, 2]):
        return "grayscale"

    scores = {
        variant: _count_colors(frame, _variant_colors(_PALETTE_PROBES, variant))
        for variant in ("default", "dark", "bright", "inverted")
    }
    return max(scores, key=scores.get)


def _variant_colors(colors: Iterable[Color], variant: ColorVariant) -> tuple[Color, ...]:
    return _cached_variant_colors(tuple(colors), variant)


@lru_cache(maxsize=None)
def _cached_variant_colors(colors: tuple[Color, ...], variant: ColorVariant) -> tuple[Color, ...]:
    return tuple({_transform_color(color, variant) for color in colors})


def _transform_color(color: Color, variant: ColorVariant) -> Color:
    values = np.asarray(color, dtype=np.float32)
    if variant == "default":
        transformed = values
    elif variant == "grayscale":
        transformed = np.repeat(np.asarray(values.mean(), dtype=np.uint8), 3)
    elif variant == "dark":
        transformed = np.clip(values * 0.55, 0, 255).astype(np.uint8)
    elif variant == "bright":
        transformed = np.clip(values * 1.35, 0, 255).astype(np.uint8)
    elif variant == "high_contrast":
        transformed = np.where(values > 127, 255, 0).astype(np.uint8)
    elif variant == "inverted":
        transformed = 255 - values
    else:
        raise ValueError(f"unknown color variant: {variant}")
    return tuple(int(value) for value in transformed)


def _transform_image(image: np.ndarray, variant: ColorVariant) -> np.ndarray:
    if variant == "default":
        return image.copy()
    if variant == "grayscale":
        gray = image.mean(axis=2, keepdims=True).astype(np.uint8)
        return np.repeat(gray, 3, axis=2)
    if variant == "dark":
        return (image.astype(np.float32) * 0.55).clip(0, 255).astype(np.uint8)
    if variant == "bright":
        return (image.astype(np.float32) * 1.35).clip(0, 255).astype(np.uint8)
    if variant == "high_contrast":
        return np.where(image > 127, 255, 0).astype(np.uint8)
    if variant == "inverted":
        return 255 - image
    raise ValueError(f"unknown color variant: {variant}")


def _classify_tile_by_template(tile: np.ndarray, x: int, y: int, variant: ColorVariant) -> Cell:
    cells, templates = _tile_templates(x, y, variant)
    distances = np.count_nonzero(templates != tile, axis=(1, 2, 3))
    return cells[int(np.argmin(distances))]


@lru_cache(maxsize=None)
def _tile_templates(x: int, y: int, variant: ColorVariant) -> tuple[tuple[Cell, ...], np.ndarray]:
    """Build renderer-shape templates; no map or runtime state is consulted."""

    height = GRID_HEIGHT * TILE_SIZE
    width = GRID_WIDTH * TILE_SIZE

    def canvas() -> np.ndarray:
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        sprite_colors.draw_floor(frame, x, y)
        return frame

    def crop(frame: np.ndarray) -> np.ndarray:
        top, left = y * TILE_SIZE, x * TILE_SIZE
        return _transform_image(frame[top : top + TILE_SIZE, left : left + TILE_SIZE], variant)

    candidates: list[tuple[Cell, np.ndarray]] = [(Cell.EMPTY, crop(canvas()))]
    static_draws = (
        (Cell.WALL, lambda frame: sprite_colors.draw_wall(frame, x, y)),
        (Cell.GAP, lambda frame: sprite_colors.draw_gap(frame, x, y)),
        (Cell.GAP, lambda frame: sprite_colors.draw_abyss(frame, x, y)),
        (Cell.BRIDGE, lambda frame: sprite_colors.draw_bridge(frame, x, y)),
        (Cell.NPC, lambda frame: sprite_colors.draw_npc(frame, x, y, COLOR_NPC)),
        (Cell.TRAP, lambda frame: sprite_colors.draw_trap(frame, x, y)),
        (Cell.BUTTON, lambda frame: sprite_colors.draw_button(frame, x, y, pressed=False)),
        (Cell.BUTTON, lambda frame: sprite_colors.draw_button(frame, x, y, pressed=True)),
        (Cell.SWITCH, lambda frame: sprite_colors.draw_switch(frame, x, y, activated=False)),
        (Cell.SWITCH, lambda frame: sprite_colors.draw_switch(frame, x, y, activated=True)),
    )
    for cell, draw in static_draws:
        frame = canvas()
        draw(frame)
        candidates.append((cell, crop(frame)))

    for opened in (False, True):
        for loot_kind in (None, "key", "gold", "heal"):
            frame = canvas()
            sprite_colors.draw_chest(frame, x, y, opened=opened, loot_kind=loot_kind)
            candidates.append((Cell.CHEST, crop(frame)))

    for facing in ("up", "down", "left", "right"):
        frame = canvas()
        sprite_colors.draw_player_sprite(frame, x * TILE_SIZE, y * TILE_SIZE, facing)
        candidates.append((Cell.PLAYER, crop(frame)))

    monster_specs = (
        ("ambusher", COLOR_MONSTER_AMBUSHER),
        ("patroller", COLOR_MONSTER_PATROLLER),
        ("chaser", COLOR_MONSTER_CHASER),
    )
    for monster_type, color in monster_specs:
        frame = canvas()
        sprite_colors.draw_monster(
            frame,
            (float(x * TILE_SIZE), float(y * TILE_SIZE)),
            TILE_SIZE,
            monster_type,
            color,
        )
        candidates.append((Cell.MONSTER, crop(frame)))

    if x in (0, GRID_WIDTH - 1) or y in (0, GRID_HEIGHT - 1):
        if x in (0, GRID_WIDTH - 1):
            exit_tile_pairs = [
                ((x, y), (x, other_y))
                for other_y in (y - 1, y + 1)
                if 0 <= other_y < GRID_HEIGHT
            ]
        else:
            exit_tile_pairs = [
                ((x, y), (other_x, y))
                for other_x in (x - 1, x + 1)
                if 0 <= other_x < GRID_WIDTH
            ]
        for exit_tiles in exit_tile_pairs:
            for exit_type, color in (
                ("normal", COLOR_EXIT_NORMAL),
                ("locked_key", COLOR_EXIT_LOCKED),
                ("conditional", COLOR_EXIT_CONDITIONAL),
            ):
                for opened in (False, True):
                    frame = canvas()
                    sprite_colors.draw_exit(frame, exit_tiles, exit_type, color, opened=opened)
                    candidates.append((Cell.EXIT, crop(frame)))

    return tuple(cell for cell, _ in candidates), np.stack([template for _, template in candidates])


def _positions(grid: np.ndarray, cell: Cell) -> tuple[GridPos, ...]:
    ys, xs = np.where(grid == int(cell))
    return tuple((int(x), int(y)) for y, x in zip(ys, xs))


def _manhattan(left: GridPos, right: GridPos) -> int:
    return abs(left[0] - right[0]) + abs(left[1] - right[1])


def _choose_nearest(
    candidates: tuple[GridPos, ...],
    previous: GridPos | None,
    *,
    max_distance: int,
) -> GridPos | None:
    if not candidates:
        return previous
    if previous is None:
        return candidates[0]
    best = min(candidates, key=lambda pos: _manhattan(pos, previous))
    if _manhattan(best, previous) <= max_distance:
        return best
    return best


def _match_positions(
    candidates: tuple[GridPos, ...],
    previous: tuple[GridPos, ...],
    *,
    max_distance: int,
) -> tuple[GridPos, ...]:
    if not previous or not candidates:
        return candidates
    remaining = list(candidates)
    matched: list[GridPos] = []
    for old in previous:
        if not remaining:
            break
        best = min(remaining, key=lambda pos: _manhattan(pos, old))
        if _manhattan(best, old) <= max_distance:
            matched.append(best)
            remaining.remove(best)
    matched.extend(remaining)
    return tuple(matched)


def _to_symbol_map(grid: np.ndarray) -> SymbolMap:
    copied = np.asarray(grid, dtype=np.uint8).copy()
    players = _positions(copied, Cell.PLAYER)
    return SymbolMap(
        grid=copied,
        player=players[0] if players else None,
        monsters=_positions(copied, Cell.MONSTER),
        chests=_positions(copied, Cell.CHEST),
        exits=_positions(copied, Cell.EXIT),
        walls=_positions(copied, Cell.WALL),
        traps=_positions(copied, Cell.TRAP),
        buttons=_positions(copied, Cell.BUTTON),
        switches=_positions(copied, Cell.SWITCH),
        npcs=_positions(copied, Cell.NPC),
        gaps=_positions(copied, Cell.GAP),
        bridges=_positions(copied, Cell.BRIDGE),
    )
