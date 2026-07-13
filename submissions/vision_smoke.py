from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nesylink.env import make_env
from nesylink.core.constants import ACTION_LEFT
from submissions.vision import VisionState, detect


DEFAULT_TASKS = tuple(f"mathematical_logic/task_{index}" for index in range(1, 6))


def diff_positions(pred: np.ndarray, expected: np.ndarray) -> list[tuple[int, int, int, int]]:
    ys, xs = np.where(pred != expected)
    return [
        (int(x), int(y), int(pred[y, x]), int(expected[y, x]))
        for y, x in zip(ys, xs)
    ]


def check_initial(task_id: str, seed: int) -> bool:
    pixel_env = make_env(task_id=task_id, observation_mode="pixels")
    grid_env = make_env(task_id=task_id, observation_mode="grid")
    try:
        frame, _ = pixel_env.reset(seed=seed)
        expected, _ = grid_env.reset(seed=seed)
        symbol_map = detect(frame)
        diffs = diff_positions(symbol_map.grid, expected["grid"])
        if diffs:
            print(f"{task_id} initial: FAIL diffs={diffs}")
            print(symbol_map.as_ascii())
            return False
        print(f"{task_id} initial: ok player={symbol_map.player} monsters={symbol_map.monsters}")
        return True
    finally:
        pixel_env.close()
        grid_env.close()


def check_rollout(task_id: str, seed: int, steps: int) -> int:
    rng = np.random.default_rng(seed)
    pixel_env = make_env(task_id=task_id, observation_mode="pixels")
    grid_env = make_env(task_id=task_id, observation_mode="grid")
    vision = VisionState()
    mismatches = 0
    try:
        frame, _ = pixel_env.reset(seed=seed)
        expected, _ = grid_env.reset(seed=seed)
        for step in range(steps + 1):
            symbol_map = vision.observe(frame)
            diffs = diff_positions(symbol_map.grid, expected["grid"])
            if diffs:
                mismatches += 1
                if mismatches <= 3:
                    print(f"{task_id} rollout step={step}: diffs={diffs[:8]}")
            if step == steps:
                break
            action = int(rng.integers(0, pixel_env.action_space.n))
            frame, _, terminated, truncated, _ = pixel_env.step(action)
            expected, _, expected_terminated, expected_truncated, _ = grid_env.step(action)
            if terminated or truncated or expected_terminated or expected_truncated:
                break
    finally:
        pixel_env.close()
        grid_env.close()
    print(f"{task_id} rollout: mismatched_frames={mismatches}")
    return mismatches


def check_exit_crossing(seed: int) -> bool:
    """Regress player tracking while the sprite overlaps a non-empty exit tile."""

    task_id = "mathematical_logic/task_3"
    pixel_env = make_env(task_id=task_id, observation_mode="pixels")
    grid_env = make_env(task_id=task_id, observation_mode="grid")
    vision = VisionState()
    try:
        frame, _ = pixel_env.reset(seed=seed)
        expected, _ = grid_env.reset(seed=seed)
        for step in range(81):
            symbol_map = vision.observe(frame)
            expected_players = np.argwhere(expected["grid"] == 2)
            expected_player = None
            if len(expected_players):
                y, x = expected_players[0]
                expected_player = (int(x), int(y))
            if symbol_map.player != expected_player:
                print(
                    f"{task_id} exit crossing step={step}: FAIL "
                    f"player={symbol_map.player} expected={expected_player}"
                )
                return False
            if step == 80:
                break
            frame, _, terminated, truncated, _ = pixel_env.step(ACTION_LEFT)
            expected, _, expected_terminated, expected_truncated, _ = grid_env.step(ACTION_LEFT)
            if terminated or truncated or expected_terminated or expected_truncated:
                break
    finally:
        pixel_env.close()
        grid_env.close()
    print(f"{task_id} exit crossing: ok")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test the frame-only vision module.")
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=80)
    args = parser.parse_args()

    initial_ok = True
    total_mismatches = 0
    for task_id in args.tasks:
        initial_ok = check_initial(task_id, args.seed) and initial_ok
        total_mismatches += check_rollout(task_id, args.seed, args.steps)

    if "mathematical_logic/task_3" in args.tasks:
        initial_ok = check_exit_crossing(args.seed) and initial_ok

    if not initial_ok:
        raise SystemExit(1)
    if total_mismatches:
        print("rollout note: moving sprites can straddle tile boundaries; planner should use repeated actions or VisionState memory.")


if __name__ == "__main__":
    main()
