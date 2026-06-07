from __future__ import annotations

from .registry import register_task
from .specs import TaskSpec


EASY_GRID_MONSTER_PERIODS = {"chaser": 1, "ambusher": 1, "patroller": 2}


BUILTIN_TASKS = (
    TaskSpec(
        task_id="task_1",
        gym_id="task_1",
        map_id="task_1",
        reward_id="collect_key",
        max_steps=500,
        mission="Collect the key and reach the exit.",
    ),
    TaskSpec(
        task_id="task_2",
        gym_id="task_2",
        map_id="task_2",
        reward_id="kill_monster",
        max_steps=500,
        mission="Defeat the monster, collect the key, and reach the exit.",
    ),
    TaskSpec(
        task_id="task_3",
        gym_id="task_3",
        map_id="task_3",
        reward_id="collect_key",
        max_steps=1000,
        mission="Travel west through the chaser room, collect the key, return, and unlock the right door.",
    ),
    TaskSpec(
        task_id="task_1_easy",
        gym_id="task_1_easy",
        map_id="task_1",
        reward_id="collect_key",
        max_steps=500,
        control_mode="grid",
        observation_mode="grid",
        monster_move_periods=EASY_GRID_MONSTER_PERIODS,
        mission="Collect the key and reach the exit with tile-level controls.",
    ),
    TaskSpec(
        task_id="task_2_easy",
        gym_id="task_2_easy",
        map_id="task_2",
        reward_id="kill_monster",
        max_steps=500,
        control_mode="grid",
        observation_mode="grid",
        monster_move_periods=EASY_GRID_MONSTER_PERIODS,
        mission="Defeat the monster, collect the key, and reach the exit with tile-level controls.",
    ),
    TaskSpec(
        task_id="task_3_easy",
        gym_id="task_3_easy",
        map_id="task_3",
        reward_id="collect_key",
        max_steps=500,
        control_mode="grid",
        observation_mode="grid",
        monster_move_periods=EASY_GRID_MONSTER_PERIODS,
        mission="Travel west, collect the key, return, and unlock the right door with tile-level controls.",
    ),
    TaskSpec(
        task_id="task_4",
        gym_id="task_4",
        map_id="task_4",
        max_steps=1000,
        control_mode="grid",
        observation_mode="grid",
        monster_move_periods=EASY_GRID_MONSTER_PERIODS,
        max_monsters=1,
        mission=(
            "Rotate the bridge to collect the key, unlock the sword room, "
            "defeat the monster, and open the revealed center chest."
        ),
        player_config={
            "items": ["shield"],
            "tools": ["shield"],
            "equipped": {"A": "none", "B": "shield"},
        },
    ),
)


def register_builtin_tasks() -> None:
    for task in BUILTIN_TASKS:
        try:
            register_task(task)
        except ValueError as exc:
            if "duplicate" not in str(exc):
                raise
