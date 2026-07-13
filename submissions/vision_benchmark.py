from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nesylink.env import make_env
from submissions.vision import Cell, VisionState
from utils.evaluate_policy import apply_obs_variant, materialize_spatial_map_variant


DEFAULT_TASKS = tuple(f"mathematical_logic/task_{index}" for index in range(1, 6))
STATIC_CELLS = {
    Cell.EMPTY,
    Cell.WALL,
    Cell.CHEST,
    Cell.EXIT,
    Cell.TRAP,
    Cell.BUTTON,
    Cell.NPC,
    Cell.GAP,
    Cell.BRIDGE,
    Cell.SWITCH,
}


@dataclass
class VisionBenchmarkResult:
    task_id: str
    seed: int
    obs_variant: str
    map_variant: str
    frames: int
    static_cells: int
    static_errors: int
    player_frames: int
    player_exact: int
    player_manhattan_sum: int
    monster_frames: int
    monster_exact: int
    monster_manhattan_sum: int

    @property
    def static_accuracy(self) -> float:
        if self.static_cells == 0:
            return 1.0
        return 1.0 - self.static_errors / self.static_cells

    @property
    def player_exact_rate(self) -> float:
        if self.player_frames == 0:
            return 1.0
        return self.player_exact / self.player_frames

    @property
    def player_avg_manhattan(self) -> float:
        if self.player_frames == 0:
            return 0.0
        return self.player_manhattan_sum / self.player_frames

    @property
    def monster_exact_rate(self) -> float:
        if self.monster_frames == 0:
            return 1.0
        return self.monster_exact / self.monster_frames

    @property
    def monster_avg_manhattan(self) -> float:
        if self.monster_frames == 0:
            return 0.0
        return self.monster_manhattan_sum / self.monster_frames


def manhattan(left: tuple[int, int], right: tuple[int, int]) -> int:
    return abs(left[0] - right[0]) + abs(left[1] - right[1])


def positions(grid: np.ndarray, cell: Cell) -> tuple[tuple[int, int], ...]:
    ys, xs = np.where(grid == int(cell))
    return tuple((int(x), int(y)) for y, x in zip(ys, xs))


def closest_total_distance(
    predicted: tuple[tuple[int, int], ...],
    expected: tuple[tuple[int, int], ...],
) -> tuple[int, int]:
    if not expected:
        return 0, 0
    remaining = list(predicted)
    exact = 0
    distance_sum = 0
    for target in expected:
        if not remaining:
            distance_sum += 99
            continue
        best = min(remaining, key=lambda pos: manhattan(pos, target))
        distance = manhattan(best, target)
        if distance == 0:
            exact += 1
        distance_sum += distance
        remaining.remove(best)
    return exact, distance_sum


def evaluate_task(
    task_id: str,
    *,
    seed: int,
    steps: int,
    obs_variant: str = "default",
    map_variant: str = "default",
) -> VisionBenchmarkResult:
    rng = np.random.default_rng(seed)
    variant_root: Path | None = None
    if map_variant == "default":
        env_kwargs = {"task_id": task_id}
    else:
        map_path = materialize_spatial_map_variant(task_id, map_variant, seed=seed)
        variant_root = map_path.parent.parent if map_path.name == "dungeon.json" else map_path.parent
        env_kwargs = {"task_id": task_id, "map_path": map_path}
    pixel_env = make_env(**env_kwargs, observation_mode="pixels")
    grid_env = make_env(**env_kwargs, observation_mode="grid")
    vision = VisionState()
    frame_count = 0
    static_cells = 0
    static_errors = 0
    player_frames = 0
    player_exact = 0
    player_manhattan_sum = 0
    monster_frames = 0
    monster_exact = 0
    monster_manhattan_sum = 0

    try:
        raw_frame, _ = pixel_env.reset(seed=seed)
        frame = apply_obs_variant(raw_frame, obs_variant)
        expected, _ = grid_env.reset(seed=seed)
        for step in range(steps + 1):
            symbol_map = vision.observe(frame)
            predicted_grid = symbol_map.grid
            expected_grid = expected["grid"]

            static_mask = np.isin(expected_grid, [int(cell) for cell in STATIC_CELLS])
            static_cells += int(static_mask.sum())
            static_errors += int((predicted_grid[static_mask] != expected_grid[static_mask]).sum())

            expected_players = positions(expected_grid, Cell.PLAYER)
            predicted_players = positions(predicted_grid, Cell.PLAYER)
            if expected_players:
                player_frames += 1
                exact, distance = closest_total_distance(predicted_players, expected_players)
                player_exact += exact
                player_manhattan_sum += distance

            expected_monsters = positions(expected_grid, Cell.MONSTER)
            predicted_monsters = positions(predicted_grid, Cell.MONSTER)
            if expected_monsters:
                monster_frames += len(expected_monsters)
                exact, distance = closest_total_distance(predicted_monsters, expected_monsters)
                monster_exact += exact
                monster_manhattan_sum += distance

            frame_count += 1
            if step == steps:
                break
            action = int(rng.integers(0, pixel_env.action_space.n))
            raw_frame, _, terminated, truncated, _ = pixel_env.step(action)
            frame = apply_obs_variant(raw_frame, obs_variant)
            expected, _, expected_terminated, expected_truncated, _ = grid_env.step(action)
            if terminated or truncated or expected_terminated or expected_truncated:
                break
    finally:
        pixel_env.close()
        grid_env.close()
        if variant_root is not None:
            shutil.rmtree(variant_root, ignore_errors=True)
            try:
                variant_root.parent.rmdir()
            except OSError:
                pass

    return VisionBenchmarkResult(
        task_id=task_id,
        seed=seed,
        obs_variant=obs_variant,
        map_variant=map_variant,
        frames=frame_count,
        static_cells=static_cells,
        static_errors=static_errors,
        player_frames=player_frames,
        player_exact=player_exact,
        player_manhattan_sum=player_manhattan_sum,
        monster_frames=monster_frames,
        monster_exact=monster_exact,
        monster_manhattan_sum=monster_manhattan_sum,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark the frame-only vision layer.")
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument(
        "--obs-variants",
        nargs="+",
        default=["default", "grayscale", "dark", "bright", "high_contrast", "inverted"],
    )
    parser.add_argument(
        "--map-variants",
        nargs="+",
        default=["default"],
        choices=["default", "spatial_a", "spatial_b", "spatial_c"],
    )
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    results = [
        evaluate_task(
            task_id,
            seed=args.seed,
            steps=args.steps,
            obs_variant=obs_variant,
            map_variant=map_variant,
        )
        for task_id in args.tasks
        for map_variant in args.map_variants
        for obs_variant in args.obs_variants
    ]
    for result in results:
        print(
            f"{result.task_id} obs={result.obs_variant} map={result.map_variant}: frames={result.frames} "
            f"static_acc={result.static_accuracy:.4f} "
            f"player_exact={result.player_exact_rate:.4f} "
            f"player_avg_dist={result.player_avg_manhattan:.3f} "
            f"monster_exact={result.monster_exact_rate:.4f} "
            f"monster_avg_dist={result.monster_avg_manhattan:.3f}"
        )

    if args.json_out is not None:
        payload = [asdict(result) | {
            "static_accuracy": result.static_accuracy,
            "player_exact_rate": result.player_exact_rate,
            "player_avg_manhattan": result.player_avg_manhattan,
            "monster_exact_rate": result.monster_exact_rate,
            "monster_avg_manhattan": result.monster_avg_manhattan,
        } for result in results]
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
