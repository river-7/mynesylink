from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nesylink.core.constants import ACTION_LABELS
from nesylink.env import make_env
from submissions.agent import TaskAgent
from utils.evaluate_policy import event_names, is_success, materialize_spatial_map_variant


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trace policy behavior on original/spatial maps.")
    parser.add_argument("--task", default="mathematical_logic/task_3")
    parser.add_argument(
        "--map-variant",
        choices=("default", "spatial_a", "spatial_b", "spatial_c"),
        default="spatial_a",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=1500)
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument("--stop-on-done", action="store_true", default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env_kwargs: dict[str, Any] = {"observation_mode": "pixels"}
    if args.map_variant == "default":
        env = make_env(task_id=args.task, **env_kwargs)
    else:
        map_path = materialize_spatial_map_variant(args.task, args.map_variant, seed=args.seed)
        env = make_env(task_id=args.task, map_path=map_path, **env_kwargs)

    agent = TaskAgent(args.task)
    captured: dict[str, Any] = {}
    original_observe = agent.vision.observe

    def observe_and_capture(frame, *, reward=None):
        symbol_map = original_observe(frame, reward=reward)
        captured["symbol_map"] = symbol_map
        return symbol_map

    agent.vision.observe = observe_and_capture  # type: ignore[method-assign]

    obs, info = env.reset(seed=args.seed)
    event_counter: Counter[str] = Counter()
    total_reward = 0.0

    print(
        f"task={args.task} map_variant={args.map_variant} seed={args.seed} "
        f"max_steps={args.max_steps}"
    )
    print(_info_line(-1, None, 0.0, info, agent, captured.get("symbol_map")))

    terminated = False
    truncated = False
    try:
        for step in range(args.max_steps):
            inventory = _inventory_from_info(info)
            action = int(agent.step(obs, inventory))
            symbol_map = captured.get("symbol_map")
            obs, reward, terminated, truncated, info = env.step(action)
            agent.update_reward(float(reward))
            total_reward += float(reward)
            step_events = event_names(info)
            event_counter.update(step_events)

            should_print = (
                step < 8
                or bool(step_events)
                or step % max(1, args.print_every) == 0
                or terminated
                or truncated
            )
            if should_print:
                print(_info_line(step, action, float(reward), info, agent, symbol_map, step_events))

            if args.stop_on_done and (terminated or truncated):
                break
    finally:
        env.close()

    print(
        "summary "
        f"success={is_success(info, terminated)} terminated={terminated} truncated={truncated} "
        f"reason={info.get('terminal_reason')} steps={info.get('episode', {}).get('step_count')} "
        f"total_reward={total_reward:.3f} events={dict(sorted(event_counter.items()))}"
    )


def _inventory_from_info(info: dict[str, Any]) -> dict[str, Any]:
    inventory = info.get("inventory", {})
    if isinstance(inventory, dict):
        return {
            "gold": inventory.get("gold"),
            "keys": inventory.get("keys"),
            "items": list(inventory.get("items", [])),
            "tools": list(inventory.get("tools", [])),
            "equipped": dict(inventory.get("equipped", {})),
        }
    return {}


def _info_line(
    step: int,
    action: int | None,
    reward: float,
    info: dict[str, Any],
    agent: TaskAgent,
    symbol_map,
    events: list[str] | None = None,
) -> str:
    action_label = "RESET" if action is None else ACTION_LABELS.get(action, str(action))
    agent_info = info.get("agent", {})
    env_info = info.get("env", {})
    inventory = info.get("inventory", {})
    debug = info.get("debug", {})
    entities = info.get("entities", {})
    player = getattr(symbol_map, "player", None)
    chests = tuple(getattr(symbol_map, "chests", ()))
    exits = tuple(getattr(symbol_map, "exits", ()))
    monsters = tuple(getattr(symbol_map, "monsters", ()))
    walls = tuple(getattr(symbol_map, "walls", ()))
    failed = getattr(agent, "_failed_tiles", {})
    blocked_exit_sides = getattr(agent, "_blocked_exit_sides_by_room", {})
    recovery = list(getattr(agent, "_recovery_actions", ()))[:8]
    recovery_labels = [ACTION_LABELS.get(action, str(action)) for action in recovery]
    opened = sorted(getattr(agent, "_opened_chests", set()))
    pending_target = getattr(agent, "_pending_chest_target", None)
    pending_open = getattr(agent, "_pending_chest_open", None)
    try:
        available = agent._available_chests(symbol_map)
    except Exception:
        available = ()
    return (
        f"step={step:04d} action={action_label:<8} reward={reward:7.3f} "
        f"room={env_info.get('room_id')} tile={agent_info.get('tile')} "
        f"vis_player={player} state={agent.state.name} goal={_current_goal(agent, symbol_map, info)} "
        f"keys={inventory.get('keys')} gold={inventory.get('gold')} "
        f"vis_chests={chests} vis_monsters={monsters} vis_exits={exits} walls={len(walls)} "
        f"move=({getattr(agent, '_move_action', None)},{getattr(agent, '_move_ticks_remaining', None)}) "
        f"recovery={recovery_labels} opened={opened} pending=({pending_target},{pending_open}) "
        f"available={available} known_exits={sorted(getattr(agent, '_known_exits', set()))} "
        f"blocked_exit_sides={blocked_exit_sides} failed={dict(failed)} entities={entities} "
        f"events={events or []} msg={debug.get('message')!r}"
    )


def _current_goal(agent: TaskAgent, symbol_map, info: dict[str, Any]) -> Any:
    if symbol_map is None:
        return None
    try:
        from submissions.fsm import get_goal

        base_goal = get_goal(agent.state, symbol_map)
        player = symbol_map.player
        if player is not None:
            exploration = agent._exploration_route_goal(symbol_map, _inventory_from_info(info), player)
            return exploration if exploration is not None else base_goal
        return base_goal
    except Exception as exc:  # pragma: no cover - debug output should not crash tracing
        return f"<goal-error {exc}>"


if __name__ == "__main__":
    main()
