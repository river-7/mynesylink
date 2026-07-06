# A Vision Notes

A's current responsibility is the frame-only perception layer:

```python
from submissions.vision import VisionState, detect

symbol_map = detect(obs)  # obs is the RGB frame, shape (128, 160, 3)

vision = VisionState()
symbol_map = vision.observe(obs, reward=last_reward)
```

The detector does not read `info`, map JSON, object coordinates, or the environment's structured `grid` observation. The structured grid is used only by `submissions/vision_smoke.py` as a local debugging oracle.

## Output Contract

`detect(frame)` returns `SymbolMap`:

- `grid`: `uint8` array with shape `(8, 10)`
- `player`: `(x, y) | None`
- `monsters`, `chests`, `exits`, `walls`, `traps`, `buttons`, `switches`, `npcs`, `gaps`, `bridges`: tuples of `(x, y)`
- `blocked_tiles()`: walls + chests + NPCs + gaps
- `danger_tiles()`: traps + monsters
- `passable_tiles(avoid_danger=True)`: candidate tiles for planner/BFS/A*

`normalize_agent_observation(obs, reward, inventory)` is available for final-policy input normalization. It accepts raw pixel arrays or README-style dict observations containing `frame`, and it only carries allowed inputs forward: frame, reward, and explicit inventory.

Cell codes match NesyLink's debug grid:

| Code | Cell |
|---:|---|
| 0 | empty |
| 1 | wall |
| 2 | player |
| 3 | monster |
| 4 | chest |
| 5 | exit |
| 6 | trap |
| 7 | button |
| 8 | npc |
| 9 | gap |
| 10 | bridge |
| 11 | switch |

## Current Method

The map is 10 by 8 cells and each cell is 16 by 16 pixels. The detector slices the RGB frame into cells and classifies each cell by exact renderer color counts. This is intentionally simple, explainable, and easy to formalize later as a symbolic abstraction assumption.

Initial frames of public Task1-Task5 currently match the environment debug grid exactly.

For moving frames, use `VisionState.observe(...)` rather than stateless `detect(...)` when integrating with a planner.

## Known Limitations

During pixel-by-pixel movement, player and monster sprites can straddle two cells. The one-frame detector may disagree with the debug grid for a few intermediate frames near tile boundaries. For planning, prefer one of these approaches:

1. Use `VisionState.observe(frame)` to keep static-map memory across frames.
2. Have the executor repeat movement actions until the player is tile-aligned before replanning.
3. Treat player/monster positions as approximate while movement is in progress.

## Local Checks

```bash
python submissions/vision_smoke.py --steps 80
python submissions/vision_benchmark.py --steps 200
```

This script runs a local oracle comparison. It is for debugging only and should not be part of the final policy's inference path.

See `docs/Mathematical_logic/A_delivery_plan.md` for the latest benchmark table and the compressed two-day delivery checklist.
