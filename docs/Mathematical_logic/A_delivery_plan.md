# A Delivery Plan

This plan compresses A's work into two days while following the assignment README requirement: final inference may use only image frames, reward history, and explicitly provided inventory/item information. `info`, map JSON, hidden object coordinates, and debug grids are allowed only for local debugging and measurement.

## Done

- Implemented `submissions/vision.py`.
- Added `detect(frame) -> SymbolMap` for one-frame perception.
- Added `VisionState.observe(frame, reward=...) -> SymbolMap` for static-map memory and dynamic-entity continuity.
- Added `normalize_agent_observation(obs, reward, inventory)` so the final policy can accept either raw pixel arrays or a README-style dict containing `frame`.
- Added local oracle checks in `submissions/vision_smoke.py`.
- Added quantitative benchmark in `submissions/vision_benchmark.py`.
- Added `submissions/vision_policy.py`, a minimal frame-only integration policy proving that the vision output can drive path planning, chest interactions, combat, and exits.
- Verified Task1, Task2, Task3, and Task4 with the frame-only policy: 3/3 seeds passed for each task using `utils/evaluate_policy.py`.
- Added task-chain memory for opened chests, known exits, monster defeat flow, key collection, Task3 west/east room traversal, and Task4 bridge-switch routing without reading `info`.

## Interface for Planner

```python
from submissions.vision import VisionState

vision = VisionState()
symbol_map = vision.observe(frame, reward=last_reward)

player = symbol_map.player
blocked = symbol_map.blocked_tiles()
danger = symbol_map.danger_tiles()
passable = symbol_map.passable_tiles(avoid_danger=True)
targets = {
    "exits": symbol_map.exits,
    "chests": symbol_map.chests,
    "monsters": symbol_map.monsters,
    "buttons": symbol_map.buttons,
    "switches": symbol_map.switches,
}
```

Coordinates are `(x, y)`, where `x` is the column `0..9`, `y` is the row `0..7`, and `grid[y, x]` is the cell code.

## Current Benchmark

Command:

```bash
python submissions/vision_benchmark.py --steps 200
```

Latest local result:

| Task | Static Accuracy | Player Exact | Player Avg Distance | Monster Exact | Monster Avg Distance |
|---|---:|---:|---:|---:|---:|
| task_1 | 0.9980 | 0.8458 | 0.154 | 1.0000 | 0.000 |
| task_2 | 0.9929 | 0.8209 | 0.179 | 0.6269 | 0.383 |
| task_3 | 0.9977 | 0.8209 | 0.179 | 1.0000 | 0.000 |
| task_4 | 0.9977 | 0.8159 | 0.184 | 1.0000 | 0.000 |
| task_5 | 0.9971 | 0.8209 | 0.179 | 0.9552 | 0.045 |

Interpretation: static geometry and interactable objects are already reliable. Player and monster exact rates are lower during pixel movement because the environment's debug grid uses entity center while the sprite can visually straddle adjacent cells. Planner/executor should replan near tile centers or treat dynamic positions as approximate one-tile obstacles.

## Remaining A Work

Day 1:

- Keep `vision_policy.py` as a regression harness: Tasks 1-4 must stay green while later behavior is added.
- Add optional tile-alignment helper for B's executor if the final planner needs a cleaner action interface.
- Save representative visual/debug screenshots if needed for report.
- Freeze `SymbolMap` interface with B.

Day 2:

- Run benchmark on several seeds and export JSON/CSV.
- Integrate with B's planner via `VisionState`.
- Extend the same frame-only memory approach toward Task5 after checking its button, trap, drain, key-door, and healing structure.
- Write final report subsection: perception assumptions, no-`info` guarantee, known limitations, benchmark table.
- Do a final check that submitted policy does not read `info` except for allowed inventory if the evaluator explicitly exposes it.

## Current Policy Smoke

```bash
python utils/evaluate_policy.py --policy submissions/vision_policy.py --tasks mathematical_logic/task_1 mathematical_logic/task_2 --num-envs 3 --max-steps 1200
python utils/evaluate_policy.py --policy submissions/vision_policy.py --tasks mathematical_logic/task_3 --num-envs 3 --max-steps 1500
python utils/evaluate_policy.py --policy submissions/vision_policy.py --tasks mathematical_logic/task_4 --num-envs 3 --max-steps 2600
```

Result:

```text
mathematical_logic/task_1
success_rate: 1.000
avg_steps:    277.0
avg_reward:   127.180

mathematical_logic/task_2
success_rate: 1.000
avg_steps:    171.0
avg_reward:   128.290

mathematical_logic/task_3
success_rate: 1.000
avg_steps:    541.0
avg_reward:   164.590
monster_killed: 1.000
key_collected: 1.000

mathematical_logic/task_4
success_rate: 1.000
avg_steps:    1110.0
avg_reward:   250.900
switch_activated: 1.000
key_collected: 1.000
door_opened: 1.000
item_collected: 1.000
monster_killed: 1.000
```

The policy still remains an A-side integration harness rather than the final team planner, but it now covers the README-constrained frame-only path through Tasks 1-4.
