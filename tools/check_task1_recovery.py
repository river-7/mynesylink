from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nesylink.core.constants import ACTION_LABELS, ACTION_UP
from submissions.agent import TaskAgent


GridPos = tuple[int, int]


@dataclass(frozen=True)
class FakeSymbolMap:
    walls: tuple[GridPos, ...]
    chests: tuple[GridPos, ...]
    monsters: tuple[GridPos, ...] = ()

    def blocked_tiles(self) -> set[GridPos]:
        return set(self.walls) | set(self.chests)


def main() -> None:
    agent = TaskAgent("mathematical_logic/task_1")
    # Matches the spatial_a failure: player at (3,3), trying UP into (3,2);
    # the failed tile has a wall on its right. Even though the player's left
    # neighbour is the opened chest at (2,3), a short in-tile LEFT nudge is the
    # useful correction; crossing a full tile left would be illegal.
    symbol_map = FakeSymbolMap(
        walls=((4, 2),),
        chests=((2, 3),),
    )
    recovery = agent._recovery_for_failed_move(symbol_map, (3, 3), ACTION_UP)
    labels = [ACTION_LABELS[action] for action in recovery[:8]]
    print(f"recovery_len={len(recovery)} first_actions={labels}")
    if not recovery or ACTION_LABELS[recovery[0]] != "LEFT":
        raise SystemExit("expected recovery to start with LEFT")


if __name__ == "__main__":
    main()
